#!/usr/bin/env bash
#
# run_pipeline.sh — run the cifar100/ pipeline on a GPU server.
#
# Replaces notebooks/kaggle_conv2net.ipynb, which is tied to /kaggle paths.
# Runs the repo's scripts unchanged; only the model selection is automated,
# via scripts/select_model.py.
#
#   ./scripts/run_pipeline.sh                      # conv2net, all steps, GPU 0
#   ./scripts/run_pipeline.sh --model vgg16 --gpu 1
#   ./scripts/run_pipeline.sh --steps 1            # train only
#   ./scripts/run_pipeline.sh --steps 5,6          # pruning only
#   ./scripts/run_pipeline.sh --setup              # create the venv and exit
#   ./scripts/run_pipeline.sh --force-train        # retrain even if a checkpoint exists
#
# --setup reads the NVIDIA driver version and installs the torch wheel built for a
# CUDA the driver actually supports. requirements.txt pins only torch>=2.0.0, so a
# plain install grabs the newest wheel and fails with "NVIDIA driver is too old".
# Override the choice with --cuda cu118|cu121|cu124|cu126|cu128|cpu.
#
# Steps follow cifar100/README.md exactly:
#   1 train, 2 activations, 3 affinity matrices, 4 PG1, 5 HRank, 6 PG pruning
# Step 3 feeds the AFC figures only and is not read by the pruner, so it is safe
# to skip with --steps 1,2,4,5,6.
#
# The README's "Running baselines" section maps to three named steps, which take
# --ratios (default 0.7, the README's value):
#
#   ./scripts/run_pipeline.sh --steps svd,sliming,snows
#   ./scripts/run_pipeline.sh --steps svd --ratios 0.5,0.7
#
# Run it under tmux. Step 6 alone is ~40-45h for conv2net.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIFAR="$REPO/cifar100"
VENV="${VENV:-$REPO/.venv}"
PY="$VENV/bin/python"

MODEL=conv2net
GPU=0
STEPS=all
RATIOS=0.7          # cifar100/README.md "Running baselines"
SETUP_ONLY=0
FORCE_TRAIN=0
CUDA_TAG=auto       # auto | cu118 | cu121 | cu124 | cu126 | cu128 | cpu

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL="$2"; shift 2 ;;
    --gpu)    GPU="$2"; shift 2 ;;
    --steps)  STEPS="$2"; shift 2 ;;
    --ratios) RATIOS="$2"; shift 2 ;;
    --venv)   VENV="$2"; PY="$VENV/bin/python"; shift 2 ;;
    --cuda)   CUDA_TAG="$2"; shift 2 ;;
    --setup)  SETUP_ONLY=1; shift ;;
    --force-train) FORCE_TRAIN=1; shift ;;
    -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' \
                   "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ "$STEPS" == "all" ]] && STEPS="1,2,3,4,5,6"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$REPO/logs/$RUN_ID"

declare -A SCRIPT=(
  [1]=train_cifar100.py
  [2]=activation_ext_cifar100.py
  [3]=cosine_sim_ext_cifar100.py
  [4]=pg_ext_cifar100.py
  [5]=hrank_cifar100.py
  [6]=pg_pruning_cifar100.py
  [svd]=svd_pruning_cifar100.py
  [sliming]=sliming_pruning_cifar100.py
  [snows]=snows_pruning_cifar100.py
)
declare -A LABEL=(
  [1]="train"                [2]="extract activations"
  [3]="affinity matrices"    [4]="PG1 extraction"
  [5]="HRank baseline"       [6]="PG / ASF-S pruning"
  [svd]="SVD baseline"       [sliming]="Sliming baseline"
  [snows]="SNOWS baseline"
)

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ── CUDA wheel selection ──────────────────────────────────────
# cifar100/requirements.txt pins only `torch>=2.0.0`, so a plain
# `pip install -r requirements.txt` fetches whatever wheel is current — which is
# built against a CUDA newer than most installed drivers and fails at runtime with
# "The NVIDIA driver on your system is too old". Pick the wheel to match the driver.
#
# The CUDA version reported by nvidia-smi is the newest runtime the driver supports.
detect_cuda_tag() {
  command -v nvidia-smi >/dev/null || { echo cpu; return; }
  local v maj min num
  v=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)
  [[ -n "$v" ]] || { echo cpu; return; }
  maj=${v%%.*}; min=${v##*.}; num=$((maj * 100 + min))
  if   (( num >= 1208 )); then echo cu128
  elif (( num >= 1206 )); then echo cu126
  elif (( num >= 1204 )); then echo cu124
  elif (( num >= 1201 )); then echo cu121
  elif (( num >= 1108 )); then echo cu118
  else echo unsupported; fi
}

# Known-good torch/torchvision pairs per wheel index. Empty = leave unpinned and
# let the index serve its newest build.
declare -A TORCH_PIN=(
  [cu118]="torch==2.4.1 torchvision==0.19.1"
  [cu121]="torch==2.5.1 torchvision==0.20.1"
  [cu124]="torch==2.6.0 torchvision==0.21.0"
  [cu126]="torch==2.7.1 torchvision==0.22.1"
  [cu128]=""
  [cpu]=""
)

setup_venv() {
  local tag="$CUDA_TAG"
  if [[ "$tag" == auto ]]; then
    tag=$(detect_cuda_tag)
    local driver
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    say "Detected driver ${driver:-none} -> wheel index: $tag"
  else
    say "Using wheel index: $tag (forced via --cuda)"
  fi

  [[ "$tag" != unsupported ]] || die "driver too old for any current PyTorch CUDA wheel.
       Ask the admin to update it, or force CPU with: $0 --setup --cuda cpu"
  [[ -v TORCH_PIN[$tag] ]] || die "unknown --cuda value '$tag'
       (valid: auto cu118 cu121 cu124 cu126 cu128 cpu)"
  [[ "$tag" != cpu ]] || warn "no GPU detected — installing CPU wheels. The pipeline
         runs but is impractically slow; step 6 would take weeks."

  say "Creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip

  # Install torch FIRST from the matching index. requirements.txt asks only for
  # torch>=2.0.0 / torchvision>=0.15.0, so the next command leaves these in place.
  say "Installing torch from the $tag index"
  # shellcheck disable=SC2086
  "$VENV/bin/pip" install ${TORCH_PIN[$tag]:-torch torchvision} \
      --index-url "https://download.pytorch.org/whl/$tag"

  say "Installing the remaining requirements"
  "$VENV/bin/pip" install -r "$CIFAR/requirements.txt"

  say "Verifying"
  "$VENV/bin/python" - <<'EOF' || die "torch cannot see the GPU — see the warning above.
       Try forcing an older wheel, e.g.: $0 --setup --cuda cu118"
import torch
print(f"  torch {torch.__version__}  built for CUDA {torch.version.cuda}  "
      f"available={torch.cuda.is_available()}")
import sys
if not torch.cuda.is_available() and torch.version.cuda:
    sys.exit(1)
EOF
  say "venv ready"
}

if [[ $SETUP_ONLY -eq 1 ]]; then setup_venv; exit 0; fi
[[ -x "$PY" ]] || die "no venv at $VENV — run: $0 --setup"

# ── preflight ─────────────────────────────────────────────────
say "Preflight"

[[ -n "${TMUX:-}" ]] || warn "not inside tmux — a dropped SSH session will kill this run.
         Start one first:  tmux new -s asfs"

"$PY" - <<'EOF' || die "torch/CUDA check failed"
import torch, sys
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  ERROR: no CUDA device visible", file=sys.stderr); sys.exit(1)
print(f"  device: {torch.cuda.get_device_name(0)}")
EOF

avail_gb=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
echo "  disk free: ${avail_gb}G"
(( avail_gb >= 15 )) || warn "under 15G free; activations alone need ~9.4G for conv2net (~25G for vgg16)"

if command -v nvidia-smi >/dev/null; then
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | wc -l)
  (( busy == 0 )) || warn "$busy process(es) already running on the GPUs — check nvidia-smi"
fi

# ── model selection ───────────────────────────────────────────
say "Selecting model: $MODEL"
"$PY" "$REPO/scripts/select_model.py" "$MODEL"

# ── warn about appending to existing results ──────────────────
for f in "$CIFAR"/pg_project_output/results/*"$MODEL"*.csv; do
  [[ -e "$f" ]] || continue
  warn "$(basename "$f") already exists. The scripts append with no header
         between runs, so new rows will land under the old ones.
         Move it aside first if you want a clean file."
done

# ── run ───────────────────────────────────────────────────────
mkdir -p "$LOGDIR"
say "Run $RUN_ID   model=$MODEL  gpu=$GPU  steps=$STEPS"
echo "  logs: $LOGDIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
cd "$CIFAR"

ckpt="$CIFAR/pg_project_output/trained_models/${MODEL}_best.pth"

IFS=',' read -ra step_list <<< "$STEPS"
for s in "${step_list[@]}"; do
  script="${SCRIPT[$s]:-}"
  [[ -n "$script" ]] || die "unknown step '$s' (valid: 1-6)"

  if [[ "$s" == "1" && -f "$ckpt" && $FORCE_TRAIN -eq 0 ]]; then
    say "Step 1 (${LABEL[1]}) — SKIPPED, checkpoint exists"
    echo "  $ckpt"
    echo "  pass --force-train to retrain"
    continue
  fi
  if [[ "$s" != "1" && ! -f "$ckpt" ]]; then
    die "step $s needs $ckpt — run step 1 first"
  fi

  # The three baselines take a CLI; the six pipeline scripts take none.
  declare -a extra=()
  case "$s" in
    svd|sliming|snows) extra=(--models "$MODEL" --ratios ${RATIOS//,/ }) ;;
  esac

  log="$LOGDIR/${s}_${script%.py}.log"
  say "Step $s — ${LABEL[$s]}  ($script ${extra[*]})"
  echo "  log: $log"
  start=$(date +%s)

  if ! "$PY" "$script" "${extra[@]}" 2>&1 | tee "$log"; then
    die "step $s failed — see $log"
  fi

  mins=$(( ($(date +%s) - start) / 60 ))
  echo "  step $s finished in ${mins} min"

  # The checkpoint is the most expensive artifact to lose; flag it early.
  if [[ "$s" == "1" ]]; then
    echo "  checkpoint written — copy it off the server now:"
    echo "    rsync -avz <user>@<host>:$ckpt ./"
  fi
done

say "Done. Results:"
ls -la "$CIFAR/pg_project_output/results/" || true
cat <<EOF

Pull results back (excluding the ~9.4G of activations):

  rsync -avz --progress --exclude 'activations/' \\
    <user>@<host>:$CIFAR/pg_project_output/ ./pg_project_output_server/
EOF
