"""Richer annotations: attributes, class hierarchy, LVIS vocabulary, per-model scorecard."""

import io
import json
import zipfile

import numpy as np

from app import class_tree, config
from app.inference import owl

from .test_api import _setup
from .test_similar import BALL, BAT, _FakeOwl, _scene


def test_attributes_round_trip_validation_and_exports(client):
    pid, label_id, image = _setup(client)
    iid = image["id"]
    ann = {"id": "a1", "label_id": label_id, "kind": "box", "geometry": {"x": 0, "y": 3, "w": 10, "h": 20},
           "source": "ai", "score": 0.9,
           "attributes": {"truncated": True, "occluded": False, "note": "behind fence",
                          "ai_suggested": ["truncated", "bogus"], "colour": "red"}}
    saved = client.put(f"/api/images/{iid}/annotations", json={"annotations": [ann]}).json()
    assert saved[0]["attributes"] == {"occluded": False, "truncated": True, "note": "behind fence",
                                      "ai_suggested": ["truncated"]}  # unknown keys dropped
    bad = {**ann, "attributes": {"occluded": "yes"}}
    assert client.put(f"/api/images/{iid}/annotations", json={"annotations": [bad]}).status_code == 422

    coco = json.loads(zipfile.ZipFile(io.BytesIO(
        client.get(f"/api/projects/{pid}/export?format=coco").content)).read("annotations.json"))
    a = coco["annotations"][0]["attributes"]
    assert a["truncated"] is True and a["note"] == "behind fence" and a["source"] == "ai"
    assert coco["categories"][0]["supercategory"] == "animal"  # label "cat": COCO supercategory

    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/api/projects/{pid}/export?format=voc").content))
    xml = zf.read(next(n for n in zf.namelist() if n.startswith("Annotations/"))).decode()
    assert "<truncated>1</truncated>" in xml and "<occluded>0</occluded>" in xml


def test_class_tree_prefers_open_images(tmp_path, monkeypatch):
    (tmp_path / "class_tree.json").write_text(json.dumps({"Man": "Person", "Cat": "Carnivore"}))
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    class_tree._open_images.cache_clear()
    try:
        assert class_tree.parent_of("man") == "Person"
        assert class_tree.parent_of("cat") == "Carnivore"  # Open Images is more specific than COCO
        assert class_tree.parent_of("tennis racket") == "sports"
        assert class_tree.parent_of("person") is None  # parent equal to itself is ignored
        assert class_tree.parent_of("spaceship") is None
    finally:
        class_tree._open_images.cache_clear()


def test_detect_vocab_names_each_object():
    patches = _scene([([20, 20, 40, 40], BALL), ([120, 120, 170, 140], BAT)])
    found = owl.detect_vocab(patches, np.stack([BALL, BAT]), ["ball", "bat"], threshold=0.5)
    assert sorted(n for n, _, _ in found) == ["ball", "bat"]
    assert owl.detect_vocab(patches, np.stack([BALL]), ["ball"], threshold=0.999) == []


class _FakeYoloEmpty:
    names = {0: "cat"}

    def detect(self, rgb, conf, iou):
        return [], 1.0


def test_detect_merges_vocab_when_asked(client, monkeypatch):
    from app.inference import registry

    _, _, image = _setup(client)
    iid = image["id"]
    monkeypatch.setattr(registry, "get_yolo", lambda: _FakeYoloEmpty())
    monkeypatch.setattr(registry, "get_owl", lambda: _FakeOwl())
    monkeypatch.setattr(registry, "get_owl_vocab", lambda: (["ball", "bat"], np.stack([BALL, BAT])))
    assert client.post(f"/api/images/{iid}/detect", json={}).json()["detections"] == []
    dets = client.post(f"/api/images/{iid}/detect", json={"vocab": True}).json()["detections"]
    assert {d["label"] for d in dets} == {"ball"} and {d["model"] for d in dets} == {"owlv2_lvis"}
    assert "owl_vocab" in client.get("/api/ai/status").json()


def test_scorecard_per_model(client):
    pid, label_id, image = _setup(client)
    iid = image["id"]
    box = {"x": 2, "y": 2, "w": 5, "h": 5}
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"id": "y1", "label_id": label_id, "kind": "box", "source": "ai", "geometry": box},
        {"id": "c1", "label_id": label_id, "kind": "box", "source": "ai_edited", "geometry": box}]})
    ev = lambda kind, ann, payload: {"kind": kind, "image_id": iid, "annotation_id": ann, "payload": payload}  # noqa: E731
    client.post(f"/api/projects/{pid}/events", json={"events": [
        ev("proposal_accepted", "y1", {"model": "yolo"}),
        ev("proposal_accepted", "c1", {"model": "yolo_coco"}),
        ev("proposal_accepted", "gone", {"model": "yolo"}),
        ev("annotation_committed", "gone", {"source": "ai", "label_id": label_id}),
        ev("annotation_deleted", "gone", {"source": "ai"})]})
    pm = client.get(f"/api/projects/{pid}/scorecard").json()["per_model"]
    assert pm["yolo"]["ai_kept"] == 1 and pm["yolo"]["ai_removed"] == 1 and pm["yolo"]["precision"] == 0.5
    assert pm["yolo_coco"]["ai_fixed"] == 1 and pm["yolo_coco"]["precision"] == 1.0
