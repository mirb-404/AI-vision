import numpy as np

from app.inference.postprocess import decode_yolo, mask_to_polygon, nms


def test_mask_to_polygon_rectangle():
    mask = np.zeros((50, 80), dtype=bool)
    mask[10:30, 20:60] = True
    poly = mask_to_polygon(mask)
    xs, ys = poly[0::2], poly[1::2]
    assert len(xs) == 4
    assert (min(xs), max(xs), min(ys), max(ys)) == (20, 59, 10, 29)


def test_mask_to_polygon_empty_and_largest():
    assert mask_to_polygon(np.zeros((10, 10), dtype=bool)) == []
    mask = np.zeros((40, 40), dtype=bool)
    mask[1:4, 1:4] = True  # small blob
    mask[10:35, 10:35] = True  # large blob wins
    xs = mask_to_polygon(mask)[0::2]
    assert min(xs) == 10


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5) == [0, 2]


def test_decode_yolo_classic_layout():
    # (1, 4 + 2 classes, 10 anchors): cx, cy, w, h, score_c0, score_c1.
    # Real exports have far more anchors than channels (84 x 8400); layout detection relies on that.
    out = np.zeros((1, 6, 10), dtype=np.float32)
    out[0, :, 0] = [50, 50, 20, 20, 0.9, 0.1]
    out[0, :, 1] = [51, 51, 20, 20, 0.8, 0.0]  # overlaps anchor 0, same class -> suppressed
    out[0, :, 2] = [51, 51, 20, 20, 0.0, 0.7]  # same place, other class -> kept
    rows = decode_yolo(out, conf=0.25, iou=0.45)
    assert rows.shape == (2, 6)
    assert rows[0].tolist() == [40, 40, 60, 60, np.float32(0.9), 0]
    assert rows[1][5] == 1


def test_decode_yolo_end_to_end_layout():
    out = np.zeros((1, 300, 6), dtype=np.float32)
    out[0, 0] = [1, 2, 3, 4, 0.8, 5]
    out[0, 1] = [1, 2, 3, 4, 0.05, 5]
    rows = decode_yolo(out, conf=0.25, iou=0.45)
    assert rows.shape == (1, 6) and rows[0][5] == 5
