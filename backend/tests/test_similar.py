"""Find alike objects (OWLv2 image-guided): query pick, scoring, threshold, box scaling, API."""

import numpy as np

from app.inference import owl

from .test_api import _setup

D = 16


def _unit(v):
    return v / np.linalg.norm(v)


def _scene(objects: list[tuple[list[float], np.ndarray]], grid: int = 20, size: float = 200.0, seed: int = 0):
    """Fake per-patch outputs: a grid of small background boxes, plus 3 patches per object whose
    boxes sit on the object (as OWLv2 predicts several boxes per object) with that object's embedding."""
    rng = np.random.default_rng(seed)
    background = _unit(np.eye(D)[1])
    step = size / grid
    boxes, embeds = [], []
    for r in range(grid):
        for c in range(grid):
            boxes.append([c * step, r * step, (c + 1) * step, (r + 1) * step])
            embeds.append(_unit(background + rng.normal(0, 0.15, D)))
    for box, emb in objects:
        for k in range(3):
            jitter = rng.normal(0, 1.0, 4)
            boxes.append(list(np.asarray(box) + jitter))
            embeds.append(_unit(emb + rng.normal(0, 0.05, D)))
    n = len(boxes)
    return {"boxes": np.asarray(boxes, dtype=np.float32), "class_embeds": np.asarray(embeds, dtype=np.float32),
            "logit_shift": np.full(n, -0.5, dtype=np.float32), "logit_scale": np.full(n, 8.0, dtype=np.float32),
            "objectness": np.zeros(n, dtype=np.float32)}


BALL = _unit(np.eye(D)[0])
OTHER_BALL = _unit(np.eye(D)[0] + 0.45 * np.eye(D)[2])  # same kind, different look (cos ~0.91)
BAT = _unit(np.eye(D)[3])  # a different kind of object


def test_select_query_takes_the_marked_object():
    src = _scene([([20, 20, 40, 40], BALL), ([120, 120, 170, 140], BAT)])
    q = owl.select_query(src, np.array([20, 20, 40, 40], dtype=np.float32))
    assert q @ BALL > 0.95 and q @ BAT < 0.3
    try:
        owl.select_query(src, np.array([500, 500, 510, 510], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("an example outside every box must be rejected")


def test_finds_same_kind_also_when_it_looks_different():
    example = np.array([20, 20, 40, 40], dtype=np.float32)
    src = _scene([(list(example), BALL)])
    q = owl.select_query(src, example)
    thr = owl.auto_threshold(src, q, example, strictness=0.5)

    tgt = _scene([([100, 30, 118, 48], BALL), ([30, 140, 60, 170], OTHER_BALL), ([140, 140, 190, 160], BAT)], seed=1)
    found = owl.detect_like(tgt, q, thr)
    centers = sorted((float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)) for b, _ in found)
    assert len(found) == 2  # one box per ball (3 predicted boxes each, merged by NMS); no bat, no background
    assert abs(centers[0][0] - 45) < 5 and abs(centers[0][1] - 155) < 5
    assert abs(centers[1][0] - 109) < 5 and abs(centers[1][1] - 39) < 5

    strict = owl.detect_like(tgt, q, owl.auto_threshold(src, q, example, strictness=1.0))
    assert len(strict) <= len(found)


class _FakeSession:
    """ONNX session stand-in returning one patch with a known normalised box."""

    def get_inputs(self):
        return [type("I", (), {"name": "pixel_values"})()]

    def get_outputs(self):
        return [type("O", (), {"name": n})() for n in
                ("class_embeds", "logit_shift", "logit_scale", "pred_boxes", "objectness")]

    def run(self, _, feed):
        assert feed["pixel_values"].shape == (1, 3, 960, 960)
        return [np.ones((1, 1, D)), np.zeros((1, 1, 1)), np.ones((1, 1, 1)),
                np.array([[[0.5, 0.25, 0.2, 0.1]]]), np.zeros((1, 1))]


def test_embed_scales_boxes_to_original_pixels(tmp_path):
    from PIL import Image
    path = tmp_path / "wide.png"
    Image.new("RGB", (400, 200), (10, 20, 30)).save(path)
    model = owl.OwlModel.__new__(owl.OwlModel)
    model.session, model.input_name, model.cache_size = _FakeSession(), "pixel_values", 2
    import threading
    from collections import OrderedDict
    model._cache, model._lock = OrderedDict(), threading.Lock()
    patches, size, _ = model.embed("k", path)
    assert size == (200, 400)
    # Padded square side = 400: cx 0.5 -> 200, cy 0.25 -> 100, w 0.2 -> 80, h 0.1 -> 40.
    assert np.allclose(patches["boxes"][0], [160, 80, 240, 120])
    assert model.embed("k", path)[2] == 0.0  # cached


class _FakeOwl:
    def __init__(self):
        self.example = np.array([2, 2, 22, 22], dtype=np.float32)

    def embed(self, key, image_path):
        # Image 64x48: the example ball at (2,2)-(22,22) and another ball at (40,5)-(60,30).
        return _scene([(list(self.example), BALL), ([40, 5, 60, 30], OTHER_BALL)], grid=8, size=64.0), (48, 64), 3.0


def test_similar_api_saves_kind_of_example_and_undo(client, monkeypatch):
    from app.inference import registry

    pid, label_id, image = _setup(client)
    iid = image["id"]
    body = {"source_image_id": iid, "box": {"x": 2, "y": 2, "w": 20, "h": 20}, "label_id": label_id}
    assert client.post(f"/api/images/{iid}/similar", json=body).status_code == 503  # no weights
    assert "owl" in client.get("/api/ai/status").json()

    monkeypatch.setattr(registry, "get_owl", lambda: _FakeOwl())
    preview = client.post(f"/api/images/{iid}/similar", json={**body, "save": False}).json()
    assert len(preview["regions"]) == 1, preview  # the other ball; the example itself is skipped
    box = preview["regions"][0]["box"]
    assert abs(box["x"] - 40) < 3 and abs(box["y"] - 5) < 3 and preview["regions"][0]["id"] is None
    assert client.get(f"/api/images/{iid}/annotations").json() == []

    saved = client.post(f"/api/images/{iid}/similar", json=body).json()  # box example -> box result
    ids = [r["id"] for r in saved["regions"]]
    anns = client.get(f"/api/images/{iid}/annotations").json()
    assert [a["id"] for a in anns] == ids and anns[0]["kind"] == "box" and anns[0]["source"] == "ai"
    assert client.post(f"/api/images/{iid}/similar", json=body).json()["regions"] == []  # no duplicates

    assert client.post("/api/similar/undo", json={"ids": ids + ["nope"]}).json() == {"deleted": 1}
    # Outline example without SAM weights: saved as an outline (the box rectangle).
    poly = client.post(f"/api/images/{iid}/similar", json={
        "source_image_id": iid, "polygon": [2, 2, 22, 2, 22, 22, 2, 22], "label_id": label_id}).json()
    ann = next(a for a in client.get(f"/api/images/{iid}/annotations").json() if a["id"] == poly["regions"][0]["id"])
    assert ann["kind"] == "polygon" and len(ann["geometry"]["points"]) == 8

    bad = client.post(f"/api/images/{iid}/similar", json={"source_image_id": iid, "label_id": label_id})
    assert bad.status_code == 422


class _FakeText:
    def encode(self, text):
        return (BALL if "ball" in text else BAT), 1.0


def test_text_search_api_finds_named_objects(client, monkeypatch):
    from app.inference import registry

    pid, label_id, image = _setup(client)
    iid = image["id"]
    body = {"text": "ball", "label_id": label_id, "strictness": 0.5}
    assert client.post(f"/api/images/{iid}/text-search", json=body).status_code == 503  # no weights
    assert "owl_text" in client.get("/api/ai/status").json()

    monkeypatch.setattr(registry, "get_owl", lambda: _FakeOwl())
    monkeypatch.setattr(registry, "get_owl_text", lambda: _FakeText())
    saved = client.post(f"/api/images/{iid}/text-search", json=body).json()
    assert len(saved["regions"]) == 2, saved  # both balls, also the one that looks different
    anns = client.get(f"/api/images/{iid}/annotations").json()
    assert {a["kind"] for a in anns} == {"box"} and len(anns) == 2
    assert client.post(f"/api/images/{iid}/text-search", json=body).json()["regions"] == []  # no duplicates
    assert client.post(f"/api/images/{iid}/text-search", json={**body, "text": "bat"}).json()["regions"] == []
    assert client.post(f"/api/images/{iid}/text-search", json={**body, "text": ""}).status_code == 422


def test_text_threshold_rises_with_strictness():
    assert owl.text_threshold(0) < owl.text_threshold(0.5) < owl.text_threshold(1)
    assert abs(owl.text_threshold(0.5) - 0.2) < 1e-9
