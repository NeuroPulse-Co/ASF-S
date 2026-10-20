#!/usr/bin/env bash
#
# run.sh – run one ASF-S step on the server and print its gate.
#
#   ./cifar10/run.sh test                              # pytest, all steps so far
#   ./cifar10/run.sh 0                                 # step 0, paper protocol, run01
#   ./cifar10/run.sh 1 --protocol clean --run-id run02
#   ./cifar10/run.sh diag_aug_parity                   # any steps/<name>.py by name
#   GPU=1 ./cifar10/run.sh 1                           # pick a GPU
#
# Extra arguments go to the step script. Logs: cifar10/outputs/<run-id>/logs/.
# Run long steps (1, 8, 9) under tmux.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${VENV:-$HERE/.venv}/bin/python"
[[ -x "$PY" ]] || { echo "no venv – run ./cifar10/setup.sh first" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

STEP="${1:?usage: run.sh <step|test> [args]}"; shift
if [[ "$STEP" == test ]]; then
  exec "$PY" -m pytest -q "$HERE/tests" "$@"
fi

SCRIPT=$(ls "$HERE"/steps/step"${STEP}"_*.py "$HERE"/steps/"${STEP}".py 2>/dev/null | head -1)
[[ -n "$SCRIPT" ]] || { echo "no script for step $STEP" >&2; exit 1; }

RUN_ID=run01
for ((i = 1; i <= $#; i++)); do
  [[ "${!i}" == --run-id ]] && { j=$((i + 1)); RUN_ID="${!j}"; }
done
LOGDIR="$HERE/outputs/$RUN_ID/logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(basename "$SCRIPT" .py)_$(date +%Y%m%d-%H%M%S).log"

echo "==> $(basename "$SCRIPT") $*   (log: $LOG)"
cd "$HERE/steps"
"$PY" -u "$SCRIPT" "$@" 2>&1 | tee "$LOG"
