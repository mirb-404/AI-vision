"""Parent class ("supercategory") of a label name, for richer exports (e.g. Man -> Person).

Two sources:
- Open Images V7 class hierarchy: models/weights/class_tree.json ({name: parent}), written on the
  Linux machine by models/export_class_tree.py from Google's published hierarchy. Optional.
- COCO supercategories (Lin et al. 2014), built in below: 80 classes in 12 groups.
Open Images is checked first because its parents are more specific.
"""

import json
from functools import lru_cache

from . import config

COCO_SUPERCATEGORIES = {
    "person": ["person"],
    "vehicle": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
    "outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
    "animal": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
    "accessory": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
               "skateboard", "surfboard", "tennis racket"],
    "kitchen": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
    "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
    "furniture": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
    "electronic": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"],
    "appliance": ["microwave", "oven", "toaster", "sink", "refrigerator"],
    "indoor": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
}
_COCO_PARENT = {name: sup for sup, names in COCO_SUPERCATEGORIES.items() for name in names}


@lru_cache(maxsize=1)
def _open_images() -> dict[str, str]:
    path = config.MODELS_DIR / "class_tree.json"
    try:
        return {k.lower(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def parent_of(name: str) -> str | None:
    """Parent class of a label name, or None when unknown (a parent equal to the name is ignored)."""
    key = name.strip().lower()
    parent = _open_images().get(key) or _COCO_PARENT.get(key)
    return parent if parent and parent.lower() != key else None
