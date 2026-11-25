# Sample images

Ten small photos for trying the app: upload them on a project page (or drag the `images/` folder in).

| Files | What | Good for |
|---|---|---|
| `images/0000000000*.jpg` (8) | Ultralytics **coco8**: 8 photos from COCO val2017 | Auto-label (YOLO), Magic select (SAM) |
| `images/bus.jpg` | Bus with 4 people | Find alike objects: mark one person, the AI marks the others |
| `images/zidane.jpg` | Two people, ties | Find alike objects, Magic select |

`labels/` holds the COCO ground-truth boxes for the 8 coco8 images in YOLO format
(`class cx cy w h`, normalised, COCO class ids 0-79). Use them to check auto-label results.
`bus.jpg` and `zidane.jpg` have no ground truth.

## Sources and licences

- coco8: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip (Ultralytics).
  The images come from COCO val2017 (Lin et al., *Microsoft COCO: Common Objects in Context*,
  ECCV 2014). COCO annotations are CC BY 4.0. The images are Flickr photos under their own
  Creative Commons licences (see https://cocodataset.org/#termsofuse).
- `bus.jpg`, `zidane.jpg`: Ultralytics example images (https://ultralytics.com/images/), as used in
  the Ultralytics documentation.

Cite COCO in the report if these images appear in it.
