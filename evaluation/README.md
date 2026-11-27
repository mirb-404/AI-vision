# evaluation/

Model-quality benchmarks that run without people. The effort study (manual vs AI-assisted) lives in
the app (Effort stats page); YOLO detection mAP lives in `training/`.

## Magic select (SAM) click benchmark: `noc_benchmark.py`

Standard interactive-segmentation protocol (RITM, Sofiiuk et al. 2021): a simulated user clicks the
centre of the object, then always the centre of the biggest mistake, up to 20 clicks.

| Metric | Meaning |
|---|---|
| NoC@85 / NoC@90 | clicks needed to reach IoU 0.85 / 0.90 (20 when never reached; lower is better) |
| mIoU@1/3/5 | mean IoU after 1, 3, 5 clicks |
| failures | images that never reach 0.85 / 0.90 in 20 clicks |
| manual vertices | clicks a hand-drawn polygon would need (outline vertices, an approximation) |

It uses the app's own SAM code (`backend/app/inference/sam.py`), so it measures what users get.

### Run (Linux machine, needs the SAM weights in `models/weights/`)

```bash
cd evaluation && mkdir -p datasets && cd datasets
# GrabCut (50 images) and Berkeley (96 images, 100 masks), as packaged by the RITM / f-BRS authors.
# Check the links on https://github.com/SamsungLabs/ritm_interactive_segmentation#datasets if they moved.
wget https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
wget https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
unzip GrabCut.zip && unzip Berkeley.zip && cd ../..

python evaluation/noc_benchmark.py --preset grabcut --root evaluation/datasets/GrabCut
python evaluation/noc_benchmark.py --preset berkeley --root evaluation/datasets/Berkeley
# any folder of images + same-named masks (non-zero = object):
python evaluation/noc_benchmark.py --images path/imgs --masks path/masks --name mine
```

Results go to `evaluation/results/<name>/`: `summary.md` (table for the report), `per_image.csv`,
`curve.csv` + `curve.png` (mean IoU per click), `metrics.json`. `datasets/` is git-ignored.

### Limits

- The previous mask is not fed back to SAM between clicks (the app does not do it either), so
  numbers can be a little worse than published SAM results.
- GrabCut and Berkeley are small and classic; SAM may have seen similar images in SA-1B.
- "Manual vertices" counts clicks, not time: placing an accurate vertex is slower than a SAM click.

## References

- Sofiiuk, Petrov, Konushin, "Reviving Iterative Training with Mask Guidance for Interactive Segmentation" (RITM), arXiv:2102.06583.
- Rother, Kolmogorov, Blake, "GrabCut", SIGGRAPH 2004 (GrabCut dataset).
- McGuinness, O'Connor, "A comparative evaluation of interactive segmentation algorithms", Pattern Recognition 2010 (Berkeley set).
- Castrejón et al., "Polygon-RNN", CVPR 2017 (click counting for polygons).
