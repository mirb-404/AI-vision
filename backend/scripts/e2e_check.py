"""End-to-end check of a running server over real HTTP.

Walks the same path a user takes in the UI: create a project and label,
upload an image, run Smart select (SAM) and Auto-label (YOLO, per image and
for the whole project), save annotations, download the annotated image, log
effort events, read stats, export COCO, YOLO and annotated images, and load
the UI page.

Run from backend/ against a server that is already up:
    python -m scripts.e2e_check http://127.0.0.1:8765 path/to/image.jpg [--no-models]

Exits non-zero on the first failed step. With --no-models the AI steps are
skipped instead of failing (for machines without exported weights).
"""

import argparse
import io
import sys
import time
import uuid
import zipfile
from pathlib import Path

import httpx


class Check:
    def __init__(self) -> None:
        self.passed = 0

    def ok(self, step: str, detail: str = "") -> None:
        self.passed += 1
        print(f"  PASS  {step}" + (f"  ({detail})" if detail else ""))

    def fail(self, step: str, detail: str) -> None:
        print(f"  FAIL  {step}: {detail}")
        sys.exit(1)

    def expect(self, step: str, resp: httpx.Response, status: int = 200) -> httpx.Response:
        if resp.status_code != status:
            self.fail(step, f"HTTP {resp.status_code} (wanted {status}): {resp.text[:300]}")
        return resp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_url")
    parser.add_argument("image", type=Path)
    parser.add_argument("--no-models", action="store_true", help="skip SAM / YOLO steps instead of failing")
    args = parser.parse_args()

    c = Check()
    api = httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120)

    print(f"End-to-end check against {args.base_url}")

    c.expect("health", api.get("/api/health"))
    c.ok("health")

    ui = c.expect("UI index", api.get("/"))
    if "<div id=\"root\"" not in ui.text and "id=root" not in ui.text:
        c.fail("UI index", "page does not look like the built UI (frontend/dist missing?)")
    c.ok("UI index", "frontend/dist served at /")

    project = c.expect("create project", api.post("/api/projects", json={"name": f"e2e-{int(time.time())}"}), 201).json()
    pid = project["id"]
    c.ok("create project", f"id {pid}")

    label = c.expect("create label", api.post(f"/api/projects/{pid}/labels", json={"name": "object"}), 201).json()
    c.ok("create label", f"id {label['id']}")

    with args.image.open("rb") as fh:
        up = c.expect("upload image", api.post(f"/api/projects/{pid}/images",
                                                files=[("files", (args.image.name, fh, "image/jpeg"))]), 201).json()
    if len(up["uploaded"]) != 1:
        c.fail("upload image", f"errors: {up['errors']}")
    image = up["uploaded"][0]
    iid, w, h = image["id"], image["width"], image["height"]
    c.ok("upload image", f"id {iid}, {w}x{h}")

    c.expect("image file", api.get(f"/api/images/{iid}/file"))
    c.ok("image file")

    status = c.expect("AI status", api.get("/api/ai/status")).json()
    c.ok("AI status", str(status))

    annotations = [{
        "id": uuid.uuid4().hex, "label_id": label["id"], "kind": "box", "source": "manual",
        "geometry": {"x": w * 0.1, "y": h * 0.1, "w": w * 0.2, "h": h * 0.2},
    }]

    if args.no_models:
        print("  SKIP  SAM embed / segment, YOLO detect (--no-models)")
    else:
        emb = c.expect("SAM embed", api.post(f"/api/images/{iid}/embed")).json()
        c.ok("SAM embed", f"{emb['latency_ms']:.0f} ms, cached={emb['cached']}")

        seg = c.expect("SAM segment (click)", api.post(f"/api/images/{iid}/segment", json={
            "points": [{"x": w / 2, "y": h / 2, "label": 1}]})).json()
        if len(seg["polygon"]) < 6:
            c.fail("SAM segment (click)", "empty polygon")
        c.ok("SAM segment (click)", f"score {seg['score']:.3f}, {len(seg['polygon']) // 2} vertices, "
                                    f"{seg['latency_ms']:.0f} ms")

        segb = c.expect("SAM segment (box)", api.post(f"/api/images/{iid}/segment", json={
            "box": {"x": w * 0.25, "y": h * 0.25, "w": w * 0.5, "h": h * 0.5}})).json()
        if len(segb["polygon"]) < 6:
            c.fail("SAM segment (box)", "empty polygon")
        c.ok("SAM segment (box)", f"score {segb['score']:.3f}")

        det = c.expect("YOLO detect", api.post(f"/api/images/{iid}/detect", json={"conf": 0.25})).json()
        if not det["detections"]:
            c.fail("YOLO detect", "no detections (use an image with common COCO objects)")
        names = ", ".join(sorted({d["label"] for d in det["detections"]}))
        c.ok("YOLO detect", f"{len(det['detections'])} detections in {det['latency_ms']:.0f} ms: {names}")

        annotations.append({"id": uuid.uuid4().hex, "label_id": label["id"], "kind": "polygon",
                            "source": "ai", "score": seg["score"], "geometry": {"points": seg["polygon"]}})
        # Auto-label only saves classes the project labels (once it has labels), so add one the
        # detector just found; 409 = it exists already.
        api.post(f"/api/projects/{pid}/labels", json={"name": det["detections"][0]["label"]})
        auto = c.expect("YOLO auto-label project", api.post(f"/api/projects/{pid}/auto-label",
                                                             json={"conf": 0.25, "skip_labeled": False})).json()
        if auto["annotations_created"] < 1:
            c.fail("YOLO auto-label project", f"nothing saved: {auto}")
        c.ok("YOLO auto-label project", f"{auto['annotations_created']} boxes saved")

        b = det["detections"][0]["box"]
        annotations.append({"id": uuid.uuid4().hex, "label_id": label["id"], "kind": "box",
                            "source": "ai", "score": det["detections"][0]["score"], "geometry": b})

    saved = c.expect("save annotations", api.put(f"/api/images/{iid}/annotations",
                                                  json={"annotations": annotations})).json()
    back = c.expect("reload annotations", api.get(f"/api/images/{iid}/annotations")).json()
    if len(saved) != len(annotations) or len(back) != len(annotations):
        c.fail("save annotations", f"sent {len(annotations)}, saved {len(saved)}, reloaded {len(back)}")
    c.ok("save + reload annotations", f"{len(back)} annotations")

    rendered = c.expect("annotated image", api.get(f"/api/images/{iid}/rendered"))
    if rendered.headers.get("content-type") != "image/jpeg":
        c.fail("annotated image", f"content-type {rendered.headers.get('content-type')}")
    c.ok("annotated image", f"{len(rendered.content) // 1024} KB JPEG")

    c.expect("mark image done", api.patch(f"/api/images/{iid}", json={"status": "done"}))
    c.ok("mark image done")

    session = c.expect("effort session", api.post(f"/api/projects/{pid}/sessions",
                                                   json={"mode": "ai", "participant": "e2e"}), 201).json()
    c.expect("log events", api.post(f"/api/projects/{pid}/events", json={"events": [
        {"session_id": session["id"], "image_id": iid, "kind": "image_opened"},
        {"session_id": session["id"], "image_id": iid, "kind": "image_done"},
    ]}), 204)
    c.ok("effort session + events", f"session {session['id']}")

    c.expect("stats", api.get(f"/api/projects/{pid}/stats"))
    c.ok("stats")
    csv = c.expect("events CSV", api.get(f"/api/projects/{pid}/events.csv")).text
    c.ok("events CSV", f"{len(csv.splitlines())} lines")

    for fmt in ("coco", "yolo", "images"):
        resp = c.expect(f"export {fmt}", api.get(f"/api/projects/{pid}/export", params={"format": fmt}))
        try:
            names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()
        except zipfile.BadZipFile:
            c.fail(f"export {fmt}", "response is not a zip")
        c.ok(f"export {fmt}", f"{len(names)} files in zip")

    print(f"\nAll {c.passed} end-to-end checks passed.")


if __name__ == "__main__":
    main()
