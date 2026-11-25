#!/usr/bin/env bash
# Full transfer-learning experiment in one command. Linux GPU machine only (Datalab).
#
#   bash training/run_all.sh
#
# Steps: venv + deps, train every variant, train the learning curve, evaluate all models
# (plus the zero-shot COCO model) on the held-out test split, write training/results/<tag>/.
#
# Options (environment variables):
#   DATA=african-wildlife.yaml   dataset YAML (name or path; app exports: see prepare_app_export.py)
#   MODEL=yolo26n                model family/scale (yolo11n also works)
#   EPOCHS=50                    epochs per run (early stopping: PATIENCE=20)
#   SEEDS="0"                    e.g. "0 1 2" to report mean +- std over seeds
#   VARIANTS="scratch head full" which variants to train
#   FRACTIONS="0.1 0.25 0.5"     learning-curve fractions for the full variant ("" to skip)
#   TAG=<dataset stem>           folder name under training/runs and training/results
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA:-african-wildlife.yaml}"
MODEL="${MODEL:-yolo26n}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-20}"
SEEDS="${SEEDS:-0}"
VARIANTS="${VARIANTS:-scratch head full}"
FRACTIONS="${FRACTIONS-0.1 0.25 0.5}"
TAG="${TAG:-$(basename "${DATA%.*}")}"

# Training is never run on the Windows dev machine (no weights there).
case "$(uname -s)" in Linux) ;; *) echo "run this on the Linux GPU machine" >&2; exit 1 ;; esac

if [ ! -d .venv-train ]; then python3 -m venv .venv-train; fi
. .venv-train/bin/activate
want="$(sha256sum training/requirements.txt | cut -d' ' -f1)"
if [ "$(cat .venv-train/.deps-stamp 2>/dev/null)" != "$want" ]; then
  pip install -q --upgrade pip
  pip install -q -r training/requirements.txt
  echo "$want" > .venv-train/.deps-stamp
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

common=(--data "$DATA" --model "$MODEL" --epochs "$EPOCHS" --patience "$PATIENCE" --tag "$TAG")
for seed in $SEEDS; do
  for v in $VARIANTS; do
    python training/train.py "${common[@]}" --variant "$v" --seed "$seed"
  done
  for f in $FRACTIONS; do
    python training/train.py "${common[@]}" --variant full --fraction "$f" --seed "$seed"
  done
done

python training/evaluate.py --data "$DATA" --split test \
  --model "zero-shot=${MODEL}.pt" --runs "training/runs/$TAG" --out "training/results/$TAG"

# Keep a copy of each run's training curves next to the results (small files, committed).
for run in training/runs/"$TAG"/*/; do
  name="$(basename "$run")"
  mkdir -p "training/results/$TAG/runs/$name"
  for f in results.csv results.png args.yaml train_config.json confusion_matrix_normalized.png; do
    if [ -f "$run/$f" ]; then cp "$run/$f" "training/results/$TAG/runs/$name/"; fi
  done
done
echo "done: training/results/$TAG/summary.md"
