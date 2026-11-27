"""Write models/weights/class_tree.json: the parent class of every Open Images box class.

Run on the Linux machine (downloads two small files from Google's Open Images site):
    python models/export_class_tree.py
Output: {"Man": "Person", "Human face": "Human body", ...}. The backend uses it to fill COCO
"supercategory" in exports (backend/app/class_tree.py). Classes directly under the root
("Entity") get no parent. Standard library only.

Source: Open Images class hierarchy (Kuznetsova et al., IJCV 2020), annotations CC BY 4.0.
"""

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path

HIERARCHY = "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json"
NAMES = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
ROOT_MID = "/m/0bl9f"  # "Entity"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def parents(node: dict, names: dict[str, str], out: dict[str, str], parent: str | None = None) -> None:
    """Walk the tree; a class can appear under several parents: the first one found is kept."""
    mid = node.get("LabelName")
    name = names.get(mid, "") if mid != ROOT_MID else ""
    if name and parent and name not in out:
        out[name] = parent
    for child in node.get("Subcategory", []):
        parents(child, names, out, name or None)
    for part in node.get("Part", []):  # parts (e.g. a wheel of a car): the whole is the parent
        parents(part, names, out, name or None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "weights" / "class_tree.json"))
    args = parser.parse_args()

    names = {row[0]: row[1] for row in csv.reader(io.StringIO(fetch(NAMES).decode("utf-8"))) if len(row) >= 2}
    tree = json.loads(fetch(HIERARCHY))
    out: dict[str, str] = {}
    parents(tree, names, out)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}: {len(out)} classes with a parent (e.g. Man -> {out.get('Man')})")


if __name__ == "__main__":
    main()
