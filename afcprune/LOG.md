# afcprune log

Decisions, runs and findings, newest first. Each run records host, commit, command and result.

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
