# afcprune log

Decisions, runs and findings, newest first. Each run records host, commit, command and result.

---

# 2026-11-24 — S1 prep: runner, smoke test on the server, dense training started

## Server state
- Host `intellisense08-EWISPro9900G` (RTX 2080). The repo is on `afc_prune` at 12e17ef. The afcprune suite
  passes there too: **73 passed** with torch 2.5.1+cu121, so the golden outputs hold across torch versions
  and devices.
- Dense checkpoints (run01, seed43, diag_tv) are backed up to the laptop under `~/asfs_backups/`.
- Clean-protocol dense models started in tmux `afcp_dense` with `cifar10/steps/step1_train.py --protocol
  clean --seed {42,43,44} --run-id afcp_dense_s{seed}`. That is the same model, data and loop code as
  afcprune's port.

## New code (uncommitted)
- `afcprune/pipeline.py`: output-layer AFC → PG1 → ΔG → PCA in one call, with the same seeded draw as
  cifar10 step 3, plus P3 diagnostics. Test: `tests/test_pipeline.py` (2 passed).
- `experiments/s1_baselines.py`: selector × budget × tier for one checkpoint, val-chosen allocation
  (BN-recalibrated val accuracy over the 40 nearest candidates), resumable CSV.
- `experiments/run_s1.sh`: phases `closed` / `gradient` / `gate`.

## Smoke run (scratch copy `~/afcp_scratch`, not the repo)
| | |
|---|---|
| Command | `s1_baselines.py --dense run01 --protocol paper --budgets t5pt,70 --selectors pgi_aligned,l1 --tiers T0,T1r,T2r,T3r,T1fc --max-epochs 1 --alloc-max 3` |
| Time | 3.1 min |
| Contrast diagnostics | dense 75.32, σ 0.831, top-4 var 0.927, min λ2−λ3 0.152, min group 50. All equal run01's step 2–5 logs, so the port matches on real data. Bootstrap PG1 median \|cos\| 0.993: PG1 is stable on CIFAR-10 (P3) |

Smoke numbers (val = test, 1 FT epoch; **not results**):

| | t5pt T1r | t5pt T3r | 70% T1r | 70% T3r |
|---|---|---|---|---|
| pgi_aligned | −11.34 | −9.94 | −8.10 | −7.72 |
| l1 | −12.72 | −10.62 | −7.58 | −7.48 |

- The closed-form ridge head alone (about 10 s) lands near run01's gradient fc FT (−10.50). The
  allocation at 70% picks 31/19, keeping conv1 almost whole, as the cifar10 grid predicted.
- Gate caveat: run01's −10.50 is a **3-seed mean**. The gate runs one FT seed with ±0.5, so seed noise
  alone could fail it. If it does, rerun with `--ft-seed 43/44` and compare the mean.

---

# 2026-11-24 — S0: skeleton, ports, protocol guard, golden regression

## Done
- Branch `afc_prune` from `origin/cifar_10_Conv2Net` (53395a6). Package `afcprune/` is self-contained.
- Ported from `cifar10/`: models, data (GPU augmentation), train loop, metrics, AFC/PG1, contrast/PCA,
  PGI (aligned and literal), surgery.
- New:
  - `protocol.py`: a guard around the test set, with an access log.
  - `allocate.py`: budget candidates with an analytic param count, and a val-chosen allocation.
  - `afc.py` additions: subspace embedding, principal angles, bootstrap PG1.
  - `scores/`: 10 criteria.
  - `recovery.py`: tiers T0–T4.
- Tests: **73 passed** (laptop, CPU, torch 2.15.0+cpu, `afcprune/.venv`).
  - Golden: PG1, λ, contrast, PCA, PGI (both modes, both models) and surgery match the original code to
    ≤1e-10 (1e-5 for logits).
  - Table 4 params (183,434 / 815,018) and FLOPs (40,894,484) exact.
  - (19, 16) kept → 75.84% params↓.

## Findings / decisions
- **run01's −10.50 is not strict head-only.** It used finetune mode `fc`, where BN running statistics
  re-adapt during FC training. Tiers are defined accordingly, and the S1 gate reproduces −10.50 in mode
  `fc`, not `fc_frozen`.
- **Conv2Net params are 89% FC** (163,850 of 183,434). Params↓ is driven by conv2 kept filters
  (2,560 FC params per conv2 filter), so per-layer allocation is the main lever. Kept counts are reported
  with every result.
- **The bootstrap catches PG1 instability.** With unstructured units (no spectral gap), resampled PG1 has
  |cos| < 0.8 to the full-sample PG1 in some draws. This is why fidelity compares top-k subspaces
  (`afc.subspace_similarity`), not v₂ alone (plan P3).
- **Related work (docs/related_work.md, agent-written, citations web-checked; recheck the arXiv IDs before
  citing).**
  - Idea 2 (closed-form head) and T2/T3 are established techniques (OBC/ThiNet/CP; EagleEye adaptive BN).
    Use them as recovery tiers and controls, not as contributions.
  - Idea 1 overlaps CKA-guided and KD-guided selection.
  - Multi-layer functional graphs + iterative pruning overlaps a 2026 spectral-distortion paper.
  - Most defensible: preserving the **correct-vs-incorrect, class-wise** relational structure (the ΔG
    contrast), and confusion-pair protection.
  - Extra controls to add in S2: a linear-CKA twin; adaptive-BN accuracy as a competing predictor; NISP; CHIP.

## Next
- S1 on the server: copy the run01 checkpoint into `$AFCP_OUTPUTS`; train 3 clean-protocol dense seeds and
  one weak (~72.7%) model; run the baseline factorial.
