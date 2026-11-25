#!/usr/bin/env bash
#
# archive_run.sh — move a finished run out of the live output folders.
#
# The pipeline scripts always write to cifar100/pg_project_output/{results,pg_data}
# and append to CSVs that already exist. Archive a run before starting the next one,
# so each run lands in its own folder under runs/ and the live folders are empty.
#
#   ./scripts/archive_run.sh run03_2026-09-30_baselines          # archive live outputs
#   ./scripts/archive_run.sh run03_... --drop-activations         # also delete activations/
#   ./scripts/archive_run.sh run03_... --dry-run                  # show what would happen
#
# Naming: runNN_<date>_<what-changed>, e.g. run02_2026-09-28_rtx2080_S50_fixes.
#
# What moves into runs/NAME/:
#   results/*   (CSVs are committed; results/pruned_models/ stays gitignored)
#   pg_data/*   (pg1_summary.csv is committed; the .npy files stay gitignored)
#   logs/*      (step logs from run_pipeline.sh; gitignored, kept on disk)
# trained_models/ is never touched: the checkpoint is shared between runs.
# activations/ is left in place unless --drop-activations (step 4 needs it to re-run).
#
# runs/NAME/ may already exist, e.g. pulled from git after archiving on the other
# machine. Files identical to what is already there are dropped from the live
# folder; a file that differs aborts the archive before anything is moved.
#
# Archive on the machine that ran the pipeline, commit there, pull elsewhere.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO/cifar100/pg_project_output"

NAME=""
DROP_ACTS=0
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --drop-activations) DROP_ACTS=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' \
                   "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "unknown option: $1" >&2; exit 2 ;;
    *)  [[ -z "$NAME" ]] || { echo "one run name only" >&2; exit 2; }; NAME="$1"; shift ;;
  esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
run()  { if [[ $DRY -eq 1 ]]; then echo "  [dry-run] $*"; else "$@"; fi; }

[[ -n "$NAME" ]] || die "usage: $0 <run-name> [--drop-activations] [--dry-run]"
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name may only contain letters, digits, . _ -"

DEST="$OUT/runs/$NAME"

# ── collect: live path → destination path ────────────────────
declare -a SRC=() DST=()
collect() {   # collect <live dir> <dest subdir>
  local dir="$1" sub="$2" f
  [[ -d "$dir" ]] || return 0
  while IFS= read -r -d '' f; do
    SRC+=("$f"); DST+=("$DEST/$sub/${f#"$dir"/}")
  done < <(find "$dir" -type f -print0 | sort -z)
}
collect "$OUT/results" results
collect "$OUT/pg_data" pg_data
collect "$REPO/logs"   logs

(( ${#SRC[@]} > 0 )) || die "nothing to archive: results/, pg_data/ and logs/ are empty"

# ── check for conflicts before touching anything ─────────────
declare -a MOVE=() SAME=()
for i in "${!SRC[@]}"; do
  if [[ -e "${DST[$i]}" ]]; then
    cmp -s "${SRC[$i]}" "${DST[$i]}" \
      || die "${DST[$i]#"$REPO"/} already exists and differs from the live copy.
       Pick another run name, or resolve by hand."
    SAME+=("$i")
  else
    MOVE+=("$i")
  fi
done

say "Archiving into ${DEST#"$REPO"/}"
echo "  move: ${#MOVE[@]} file(s)    identical to archived copy, drop: ${#SAME[@]}"
for i in "${MOVE[@]}"; do
  run mkdir -p "$(dirname "${DST[$i]}")"
  run mv "${SRC[$i]}" "${DST[$i]}"
done
for i in "${SAME[@]}"; do run rm "${SRC[$i]}"; done

# empty subfolders left behind (e.g. results/pruned_models/pg_conv2net)
for d in "$OUT/results" "$OUT/pg_data" "$REPO/logs"; do
  [[ -d "$d" ]] && run find "$d" -mindepth 1 -type d -empty -delete
done

if [[ $DROP_ACTS -eq 1 && -d "$OUT/activations" ]]; then
  echo "  deleting activations/ ($(du -sh "$OUT/activations" | cut -f1))"
  run find "$OUT/activations" -mindepth 1 -delete
fi

# ── RUN_INFO.md: machine-written facts, regenerated on every archive ──
info() {
  local py="$REPO/.venv/bin/python"
  echo "# Run info — $NAME"
  echo
  echo "Written by \`scripts/archive_run.sh\`. Put the analysis in README.md."
  echo
  echo "| | |"
  echo "|---|---|"
  echo "| Archived | $(date '+%Y-%m-%d %H:%M %Z') |"
  echo "| Host | $(hostname) |"
  echo "| Git | \`$(git -C "$REPO" rev-parse --short HEAD)\` on \`$(git -C "$REPO" branch --show-current)\`$(
          git -C "$REPO" diff --quiet -- cifar100 scripts || echo ' (uncommitted code changes)') |"
  if command -v nvidia-smi >/dev/null; then
    echo "| GPU | $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) |"
  fi
  [[ -x "$py" ]] && echo "| torch | $("$py" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '?') |"
  echo
  echo "## Files"
  echo
  echo "| File | Rows |"
  echo "|---|---|"
  local f
  while IFS= read -r f; do
    echo "| \`${f#"$DEST"/}\` | $(( $(wc -l < "$f") - 1 )) |"
  done < <(find "$DEST" -name '*.csv' | sort)
  if [[ -d "$DEST/logs" ]]; then
    echo
    echo "## Step logs (gitignored, on disk only)"
    echo
    find "$DEST/logs" -name '*.log' | sort | sed "s|^$DEST/|- \`|; s|$|\`|"
  fi
}

if [[ $DRY -eq 0 ]]; then
  info > "$DEST/RUN_INFO.md"
  if [[ ! -e "$DEST/README.md" ]]; then
    cat > "$DEST/README.md" <<EOF
# ${NAME}

<!-- What was run, why, what changed since the previous run, and what it showed.
     Machine facts (host, commit, GPU, file list) are in RUN_INFO.md. -->
EOF
  fi
fi

if [[ $DRY -eq 1 ]]; then say "Dry run: nothing was moved"; exit 0; fi
say "Done"
echo "  live results/ and pg_data/ are now empty; trained_models/ untouched"
echo "  next:  edit ${DEST#"$REPO"/}/README.md, then"
echo "         git add ${DEST#"$REPO"/} && git commit"
