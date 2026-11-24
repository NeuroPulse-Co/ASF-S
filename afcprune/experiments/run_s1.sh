#!/usr/bin/env bash
#
# run_s1.sh – S1 baseline factorial on the server, one dense checkpoint after another.
#
#   experiments/run_s1.sh closed     # T0, T1r, T2r, T3r for every selector × budget   (~1.7 h / model)
#   experiments/run_s1.sh gradient   # T1g, T1fc, T4                                    (~10 h / model)
#   experiments/run_s1.sh gate       # run01 reproduction gate only: pgi_aligned @ t5pt, T1fc
#
# Dense checkpoints are read from $ASFS_OUTPUTS (cifar10 step 1 runs); results go to $AFCP_OUTPUTS/s1/<tag>/.
# Rows already in results.csv are skipped, so the script can be re-run after an interruption.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$HERE/../cifar10/.venv/bin/python}"
export AFCP_DATA="${AFCP_DATA:-$HERE/../cifar10/data}"
export AFCP_OUTPUTS="${AFCP_OUTPUTS:-$HOME/afcp_outputs}"
DENSE="${ASFS_OUTPUTS:-$HOME/asfs_outputs}"

# tag  protocol  checkpoint
MODELS=(
  "run01_paper paper $DENSE/run01/conv2net_paper/step1_train/model_best.pth"
  "clean_s42   clean $DENSE/afcp_dense_s42/conv2net_clean/step1_train/model_best.pth"
  "clean_s43   clean $DENSE/afcp_dense_s43/conv2net_clean/step1_train/model_best.pth"
  "clean_s44   clean $DENSE/afcp_dense_s44/conv2net_clean/step1_train/model_best.pth"
)

phase="${1:?usage: run_s1.sh closed|gradient|gate}"
mkdir -p "$AFCP_OUTPUTS/logs"
for m in "${MODELS[@]}"; do
  read -r tag proto ckpt <<<"$m"
  [[ -f "$ckpt" ]] || { echo "skip $tag: no checkpoint $ckpt"; continue; }
  case "$phase" in
    closed)   extra=(--tiers T0,T1r,T2r,T3r) ;;
    gradient) extra=(--tiers T1g,T1fc,T4) ;;
    gate)     [[ "$proto" == paper ]] || continue
              extra=(--tiers T1fc --selectors pgi_aligned --budgets t5pt) ;;
    *) echo "unknown phase $phase"; exit 1 ;;
  esac
  echo "== $tag ($proto) $phase  $(date '+%F %T')"
  "$PY" -u "$HERE/experiments/s1_baselines.py" --dense "$ckpt" --tag "$tag" --protocol "$proto" \
      "${extra[@]}" 2>&1 | grep --line-buffered -v "Warning" | tee -a "$AFCP_OUTPUTS/logs/s1_${tag}_${phase}.log"
done
