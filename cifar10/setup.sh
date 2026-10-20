#!/usr/bin/env bash
#
# setup.sh – create cifar10/.venv with a torch wheel the server's NVIDIA driver supports.
#
#   ./cifar10/setup.sh                 # detect from nvidia-smi
#   ./cifar10/setup.sh --cuda cu121    # force a wheel index (cu118|cu121|cu124|cu126|cu128|cpu)
#
# Same wheel logic as scripts/run_pipeline.sh: the RTX 2080 server (driver 535, CUDA 12.2)
# gets torch 2.5.1+cu121, the version runs 02/03 used.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$HERE/.venv}"
CUDA_TAG=auto
[[ "${1:-}" == "--cuda" ]] && CUDA_TAG="$2"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

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

declare -A TORCH_PIN=(
  [cu118]="torch==2.4.1 torchvision==0.19.1"
  [cu121]="torch==2.5.1 torchvision==0.20.1"
  [cu124]="torch==2.6.0 torchvision==0.21.0"
  [cu126]="torch==2.7.1 torchvision==0.22.1"
  [cu128]="torch torchvision"
  [cpu]="torch torchvision"
)

[[ "$CUDA_TAG" == auto ]] && CUDA_TAG=$(detect_cuda_tag)
[[ "$CUDA_TAG" != unsupported ]] || die "driver too old for current PyTorch CUDA wheels"
[[ -v TORCH_PIN[$CUDA_TAG] ]] || die "unknown --cuda value '$CUDA_TAG'"
say "Wheel index: $CUDA_TAG"

python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
# shellcheck disable=SC2086
"$VENV/bin/pip" install ${TORCH_PIN[$CUDA_TAG]} --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

"$VENV/bin/python" - <<'PY'
import sys, torch
print(f"  torch {torch.__version__}  CUDA {torch.version.cuda}  available={torch.cuda.is_available()}")
if torch.version.cuda and not torch.cuda.is_available():
    sys.exit("torch was built for CUDA but cannot see the GPU")
PY
say "venv ready: $VENV"
