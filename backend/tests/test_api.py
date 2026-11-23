import io
import json
import zipfile

from PIL import Image


def _png(w=64, h=48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _setup(client):
    project = client.post("/api/projects", json={"name": "Test project"}).json()
    pid = project["id"]
    label = client.post(f"/api/projects/{pid}/labels", json={"name": "cat", "color": "#ff0000"}).json()
    res = client.post(
        f"/api/projects/{pid}/images",
        files=[("files", ("a.png", _png(), "image/png")), ("files", ("bad.txt", b"not an image", "text/plain"))],
    )
    assert res.status_code == 201
    body = res.json()
    assert len(body["uploaded"]) == 1 and len(body["errors"]) == 1
    return pid, label["id"], body["uploaded"][0]


def test_full_flow(client):
    pid, label_id, image = _setup(client)
    assert (image["width"], image["height"]) == (64, 48)
    iid = image["id"]

    assert client.get(f"/api/images/{iid}/file").status_code == 200

    anns = [
        {"id": "b1", "label_id": label_id, "kind": "box", "geometry": {"x": 2, "y": 3, "w": 10, "h": 20}},
        {"id": "p1", "label_id": label_id, "kind": "polygon", "source": "ai", "score": 0.9,
         "geometry": {"points": [10, 10, 30, 10, 30, 30]}},
    ]
    res = client.put(f"/api/images/{iid}/annotations", json={"annotations": anns})
    assert res.status_code == 200, res.text
    assert len(res.json()) == 2
    assert client.get(f"/api/images/{iid}").json()["status"] == "in_progress"

    # Replace: drop the box, keep the polygon.
    res = client.put(f"/api/images/{iid}/annotations", json={"annotations": anns[1:]})
    assert [a["id"] for a in res.json()] == ["p1"]

    # Label in use cannot be deleted.
    assert client.delete(f"/api/labels/{label_id}").status_code == 409

    # COCO export
    res = client.get(f"/api/projects/{pid}/export?format=coco")
    assert res.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    coco = json.loads(zf.read("annotations.json"))
    assert len(coco["images"]) == 1 and len(coco["annotations"]) == 1
    ann = coco["annotations"][0]
    assert ann["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert ann["area"] == 200.0
    assert coco["images"][0]["file_name"] in zf.namelist()

    # YOLO export
    res = client.get(f"/api/projects/{pid}/export?format=yolo")
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    label_files = [n for n in zf.namelist() if n.startswith("labels/")]
    line = zf.read(label_files[0]).decode().strip()
    cls, cx, cy, w, h = line.split()
    assert cls == "0" and abs(float(cx) - 20 / 64) < 1e-5 and abs(float(h) - 20 / 48) < 1e-5


def test_validation(client):
    pid, label_id, image = _setup(client)
    iid = image["id"]
    bad_poly = {"id": "x", "label_id": label_id, "kind": "polygon", "geometry": {"points": [1, 2, 3, 4]}}
    assert client.put(f"/api/images/{iid}/annotations", json={"annotations": [bad_poly]}).status_code == 422
    other = client.post("/api/projects", json={"name": "Other"}).json()
    foreign = client.post(f"/api/projects/{other['id']}/labels", json={"name": "dog"}).json()
    ok_box = {"id": "y", "label_id": foreign["id"], "kind": "box", "geometry": {"x": 0, "y": 0, "w": 1, "h": 1}}
    assert client.put(f"/api/images/{iid}/annotations", json={"annotations": [ok_box]}).status_code == 422
    assert client.post(f"/api/projects/{pid}/labels", json={"name": "CAT"}).status_code == 409


def test_ai_without_models(client):
    pid, _, image = _setup(client)
    status = client.get("/api/ai/status").json()
    assert status["sam"]["available"] is False and status["yolo"]["available"] is False
    res = client.post(f"/api/images/{image['id']}/segment", json={"points": [{"x": 5, "y": 5, "label": 1}]})
    assert res.status_code == 503
    assert client.post(f"/api/images/{image['id']}/detect", json={}).status_code == 503


def test_events_and_stats(client):
    pid, _, image = _setup(client)
    manual = client.post(f"/api/projects/{pid}/sessions", json={"mode": "manual"}).json()
    ai = client.post(f"/api/projects/{pid}/sessions", json={"mode": "ai", "participant": "p1"}).json()

    def committed(sid, ms, clicks, tool):
        return {"session_id": sid, "image_id": image["id"], "kind": "annotation_committed",
                "payload": {"duration_ms": ms, "clicks": clicks, "tool": tool}}

    events = [committed(manual["id"], 20000, 12, "polygon"), committed(manual["id"], 30000, 14, "polygon"),
              committed(ai["id"], 5000, 2, "smart"), committed(ai["id"], 10000, 3, "smart"),
              {"session_id": ai["id"], "kind": "proposal_accepted"},
              {"session_id": ai["id"], "kind": "proposal_rejected"}]
    assert client.post(f"/api/projects/{pid}/events", json={"events": events}).status_code == 204

    stats = client.get(f"/api/projects/{pid}/stats").json()
    assert stats["per_mode"]["manual"]["seconds_per_object"]["median"] == 25.0
    assert stats["per_mode"]["ai"]["seconds_per_object"]["median"] == 7.5
    assert abs(stats["speed_up_median"] - 25 / 7.5) < 1e-9
    assert stats["proposals"]["accept_rate"] == 0.5

    csv_text = client.get(f"/api/projects/{pid}/events.csv").text
    assert csv_text.count("annotation_committed") == 4

    # A session from another project is rejected.
    other = client.post("/api/projects", json={"name": "O"}).json()
    res = client.post(f"/api/projects/{other['id']}/events", json={"events": [{"session_id": ai["id"], "kind": "image_opened"}]})
    assert res.status_code == 422
