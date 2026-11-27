"""Click simulator of evaluation/noc_benchmark.py (RITM protocol)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
import noc_benchmark as nb  # noqa: E402


def test_first_click_is_object_centre_and_noc_counts():
    gt = np.zeros((40, 40), bool)
    gt[10:30, 5:25] = True
    valid = np.ones_like(gt)
    x, y, label = nb.next_click(np.zeros_like(gt), gt, valid)
    assert label == 1 and 13 <= x <= 16 and 18 <= y <= 21

    pred = np.ones_like(gt)  # covers far too much: the next click must be a background click
    x, y, label = nb.next_click(pred, gt, valid)
    assert label == 0 and not gt[int(y), int(x)]

    assert nb.noc([0.5, 0.86, 0.91], 0.85) == 2 and nb.noc([0.5, 0.86, 0.91], 0.90) == 3
    assert nb.noc([0.1] * 20, 0.85) == nb.MAX_CLICKS
    assert nb.iou(gt, gt, valid) == 1.0
    assert nb.manual_vertices(gt) == 4  # a rectangle needs 4 polygon clicks
