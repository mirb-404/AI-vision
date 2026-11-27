"""Merging two detectors: broad Open Images model + COCO model."""

import numpy as np

from app.inference.yolo import MergedDetector, merge_detections


def _d(label, score, x, y, w, h):
    return {"label": label, "class_id": 0, "score": score, "box": {"x": x, "y": y, "w": w, "h": h}}


def test_merge_fills_misses_drops_scenery_and_duplicates():
    area = 100.0 * 100.0
    broad = [_d("Tree", 0.6, 0, 0, 100, 95),      # scenery over the whole image: dropped
             _d("Man", 0.7, 10, 10, 30, 60),
             _d("Human face", 0.8, 18, 12, 10, 10)]  # inside the man, same model: kept
    coco = [_d("person", 0.9, 11, 10, 30, 61),      # same object as "Man": the more confident wins
            _d("elephant", 0.5, 50, 40, 40, 40)]    # missed by the broad model: added
    labels = [d["label"] for d in merge_detections([broad, coco], area)]
    assert labels == ["person", "Human face", "elephant"]


def test_single_model_keeps_big_boxes():
    # With one detector nothing fills in for it, so its big boxes stay.
    assert len(merge_detections([[_d("bus", 0.9, 0, 0, 100, 100)]], 100.0 * 100.0)) == 1


class _Fake:
    def __init__(self, tag, names, dets):
        self.tag, self.names, self._dets = tag, names, dets

    def detect(self, rgb, conf, iou):
        return [dict(d, box=dict(d["box"])) for d in self._dets], 2.0


def test_merged_detector_tags_models_and_offsets_class_ids():
    a = _Fake("yolo", {0: "Man", 1: "Tree"}, [_d("Man", 0.7, 0, 0, 10, 10)])
    b = _Fake("yolo_coco", {0: "person", 1: "elephant"}, [dict(_d("elephant", 0.6, 50, 50, 20, 20), class_id=1)])
    m = MergedDetector([a, b], ["yolo", "yolo_coco"])  # type: ignore[list-item]
    assert m.names == {0: "Man", 1: "Tree", 2: "person", 3: "elephant"}
    dets, ms = m.detect(np.zeros((100, 100, 3), np.uint8), 0.1, 0.45)
    assert ms == 4.0 and [(d["label"], d["model"], d["class_id"]) for d in dets] == [
        ("Man", "yolo", 0), ("elephant", "yolo_coco", 3)]
