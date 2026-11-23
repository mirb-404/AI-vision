"""Print the inputs, outputs and metadata of ONNX files.

    python models/inspect_onnx.py models/weights/*.onnx

Use it to confirm an export matches what backend/app/inference expects
(see the I/O contract in models/README.md).
"""

import sys

import onnxruntime as ort


def main(paths: list[str]) -> None:
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        print(f"== {path}")
        for i in sess.get_inputs():
            print(f"  in   {i.name:24s} {i.type:16s} {i.shape}")
        for o in sess.get_outputs():
            print(f"  out  {o.name:24s} {o.type:16s} {o.shape}")
        meta = sess.get_modelmeta().custom_metadata_map
        if meta:
            print(f"  meta {', '.join(sorted(meta))}")


if __name__ == "__main__":
    main(sys.argv[1:])
