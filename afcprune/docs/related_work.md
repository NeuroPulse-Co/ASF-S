# Related work and novelty check (plan item P12)

Purpose: before any afcprune idea is presented as new, place it against the closest prior work and list the
control experiments each overlap requires. Written 2026-11-24.

Citation status:
- **✓** = venue and year checked online in this pass.
- **(std)** = a standard citation, not re-checked online in this pass.
- **[unverified]** = could not be confirmed.

## Our ideas, in brief
- **I1, AFC fidelity selection.** Choose which filters to keep by how well the pruned net preserves the dense
  net's output-layer functional connectivity. That connectivity is the per-class, correct-vs-incorrect cosine →
  Gaussian → diffusion-map embedding, compared as a subspace.
- **I2, closed-form head.** Refit only the output layer by ridge regression on the kept features (no gradients).
- **S4.1, multi-layer AFC.** Build the FC graph over *hidden* units (channels), not only the 10 logits.
- **S4.2, FC redundancy / diversity.** Keep a set of units that covers the FC graph: cluster representatives, or
  max-volume / Nyström column selection.
- **S4.3, iterative rounds.** Re-estimate AFC and BN after each pruning round.
- **S4.4, allocation across layers.** Split the pruning budget across layers using per-layer sensitivity curves
  measured on val.
- **S4.5, class-conditional scoring.** Protect the units behind specific confusions.

## Closest prior work

| Method | What it does | Overlap with us | What would make ours distinct |
|---|---|---|---|
| **ThiNet** (Luo, Wu, Lin, ICCV 2017) (std) | Prunes channel *i* of layer L by greedily choosing the input channels of L+1 that best reconstruct L+1's output (least squares), then rescales the weights | I1's search (greedy, output-preserving) and our T3 compensation are the same mechanism | I1 differs only in *what* is preserved: the class-conditional, correct/incorrect *relational* structure of the outputs, not their values. That counts only if it beats a ThiNet-style reconstruction twin |
| **Channel Pruning** (He, Zhang, Sun, ICCV 2017) (std) | LASSO channel selection + least-squares reconstruction of the next layer's output | Same as ThiNet. T3 is essentially its reconstruction step | T3 is a recovery tier, not a contribution. Cite it as such |
| **NISP** (Yu et al., CVPR 2018) ✓ | Gets importance at the final response layer, then propagates it back to every neuron in one backward pass | The idea "importance is defined at the output and pushed back to earlier layers" is exactly ASF-S's premise. Paper PGI-aligned ≈ a linear NISP | The output-side signal is relational (connectivity among output units, conditioned on correctness), not per-neuron magnitude. Needs a NISP twin |
| **CHIP** (Sui et al., NeurIPS 2021) ✓ | Prunes channels whose feature maps are *least independent* of the others (nuclear-norm change when a channel is removed); inter-channel, data-driven | S4.1 + S4.2 (hidden-unit FC + redundancy) overlap strongly: both score a channel by its relation to the other channels' activations | Ours uses a graph/diffusion embedding and is class- and correctness-conditional, and it is tied to preserving output AFC. Without that coupling, S4.2 is a CHIP variant |
| **Central Filter** (Duan et al., arXiv 2112.05493, 2021; venue [unverified]) ✓ | Builds a similarity graph on feature maps, keeps the most central filter (closeness centrality) and folds the others into the next layer | S4.2 (keep graph representatives) + T3 almost exactly | Same as CHIP: the distinguishing part must come from the class/correctness conditioning and the diffusion embedding |
| **ACSP** (arXiv 2505.13225, 2025) ✓ | Graph built from activations, then clustering to select a diverse, complementary subset of units per layer | S4.2 directly | Same as above |
| **HRank** (Lin et al., CVPR 2020) (std) | Keeps filters whose feature maps have high average rank; intra-channel, label-free | Both are activation-based. The paper's own text positions ASF-S against HRank | We already plan HRank as a baseline |
| **FPGM** (He et al., CVPR 2019) (std) | Prunes filters near the geometric median of their layer (weight-space redundancy) | S4.2's redundancy idea, but in weight space | Ours measures redundancy in function (activations) space. FPGM is already a planned baseline |
| **Taylor** (Molchanov et al., ICLR 2017; CVPR 2019) (std) | First-order estimate of the loss change from removing a filter | I1(c) (gradient-of-fidelity × mask) is a Taylor expansion of a different objective | Fair comparison: Taylor on the same objective. A **Taylor-on-logit-KL** twin isolates the objective |
| **Network Slimming** (Liu et al., ICCV 2017) (std) | Ranks channels by BN γ (with L1-sparsified γ during training) | — | We use BN-γ without sparsity training as a baseline only |
| **OBS / OBC / SparseGPT** (Hassibi & Stork, NeurIPS 1992 (std); Frantar, Singh, Alistarh, NeurIPS 2022 ✓; Frantar & Alistarh, ICML 2023 (std)) | Second-order, layer-wise, post-training removal plus closed-form weight update from a small calibration set | I2 and T3 are closed-form, calibration-only recovery, which is the OBC setting. OBC also shows that "no retraining" post-training pruning is an established line | I2 cannot be claimed as novel. It is a recovery tier and a required control (P2) |
| **Neuron Merging** (Kim et al., NeurIPS 2020) ✓ | Folds pruned neurons into similar kept ones (cosine similarity) via a scaling absorbed into the next layer; data-free, no FT | T3 / S4.2 (merge redundant units instead of dropping them) | Use it as an alternative T3 compensation. Not our contribution |
| **KD-guided pruning** (Minitron, arXiv 2408.11796 ✓; KDFS, arXiv 2307.00198 ✓) | Teacher-logit KL mostly for *recovery*. KDFS uses teacher features to guide filter sampling | I1's twin (logit-KL selection) is this idea used for selection | Our planned logit-KL twin covers it |
| **EagleEye / adaptive BN** (Li et al., ECCV 2020) ✓ | Recalibrates BN statistics on train batches to evaluate pruned candidates; shows strong correlation with post-FT accuracy | Our T2 tier, and the S2 "does fidelity predict accuracy?" design | T2 is known. S2 must also correlate *adaptive-BN accuracy* (EagleEye's proxy) with final accuracy; AFC fidelity must beat it to be useful as a candidate evaluator |
| **Spectral graph sparsification** (Laenen, ICML 2023 TAG-ML workshop, PMLR 221) ✓ | One-shot pruning that preserves the spectrum of the network's *weight* graph | Spectral preservation | Ours preserves the spectrum of an *activation* (functional) graph. Different object, same spirit. Cite it |
| **Spectral structural distortion** (Y. Wang, arXiv 2605.18860, May 2026; preprint) ✓ | Scores each neuron by its contribution to spectral distortion between a layer's input- and output-side relational graphs built from hidden states; iterative removal; CNNs, encoders, LLMs | **The closest match to S4.1 + S4.3** (multi-layer functional graph, spectral preservation, iterative) | Our distinct parts: the class × correct/incorrect conditioning, the diffusion-map (not Laplacian) embedding, and anchoring at the output AFC. This must be a compared or discussed baseline, otherwise S4.1/S4.3 look derivative |
| **Functional connectomes of NNs** (Songdechakraiwut & Wu, AAAI 2025) ✓ | Builds FC graphs from neuron activations and analyses them with persistent graph homology, for interpretability (not pruning) | Supports the AFC framing | Cite it as motivation. It does not prune |
| **CKA / RSA** (Kornblith et al., ICML 2019 (std); Kriegeskorte et al., 2008 (std)) | Representation similarity between networks or layers | AFC fidelity is a representation-similarity measure. CKA is the default one | Needs a **CKA-preservation twin**: the same search, objective = linear CKA of output (or hidden) representations |
| **CKA-guided pruning** (Pons et al., arXiv 2405.17081, 2024 ✓; arXiv 2506.04513, 2025 ✓) | Layer removal by CKA to the original; 2506.04513 alternates layer and filter pruning and picks the candidate with higher representation similarity to its parent | I1's selection principle ("keep the subnetwork most similar to its parent") is already published, with CKA | Strongest novelty threat to I1. I1 is new only if the *class/correctness-conditional diffusion geometry* beats CKA under the same search |
| **Class-aware pruning** (Jiang et al., arXiv 2312.05875 ✓; CATRO, Hu et al., arXiv 2110.10921 ✓) | Score filters by how many classes they serve (class-aware), or by joint class discrimination of the kept channels (CATRO) | S4.5 | Ours targets confusion pairs specifically (via the correct-vs-incorrect contrast), which neither does explicitly (per their abstracts). A CATRO baseline is needed if S4.5 is pursued |

The seed paper itself now appears on arXiv as 2609.24401 ("Artificial Structure Function Search") ✓ (found in search
results, not opened). Cite the arXiv version rather than the anonymous NeurIPS submission.

## (a) Controls implied by the overlaps

Each idea is run against twins that use the same search, budget, allocation procedure, recovery tier and seeds.
Only the objective or criterion changes.

| Our idea | Required twins / baselines |
|---|---|
| I1 AFC-fidelity selection | **logit-KL twin** (KD), **logit-MSE / ThiNet-reconstruction twin**, **linear-CKA twin** (Pons 2024 / 2506.04513), **adaptive-BN-accuracy twin** (EagleEye), NISP |
| I1(c) first-order variant | **Taylor on logit-KL** (Molchanov objective swap) |
| I2 closed-form head, T3 | Not a contribution. Report as tiers for *every* selector (P2). T3 alternatives: ThiNet LS vs Neuron Merging |
| S2 "does AFC predict accuracy?" | Add CKA, logit-KL and **adaptive-BN accuracy** as competing predictors |
| S4.1 multi-layer AFC | **Spectral-structural-distortion (Wang 2026)**, CHIP, HRank |
| S4.2 FC redundancy / diversity | **CHIP**, **Central Filter**, ACSP, FPGM (weight-space counterpart) |
| S4.3 iterative | The same method one-shot vs iterative, with an equal compute budget for every baseline's iterative variant |
| S4.5 class-conditional | **CATRO**, Class-Aware Pruning |

## (b) Verdict

- **Not novel; present as tiers or controls:** I2 (closed-form head, OBC/ThiNet line), T2 (adaptive BN,
  EagleEye), T3 (ThiNet / Channel Pruning / Neuron Merging), iterative pruning on its own, val-based allocation.
- **At risk:** I1 as "keep the subnetwork whose representation is most similar to the dense one" is already
  covered by CKA-guided pruning (2024–25) and KD-guided selection. S4.1 + S4.3 are close to Wang (2026), and
  S4.2 is close to CHIP / Central Filter / ACSP.
- **Most defensible novelty:** the *conditioning*, i.e. preserving the **class-wise correct-vs-incorrect
  relational structure** (the ΔG contrast) rather than the representation as a whole. Two concrete forms:
  1. I1 with the contrast-subspace objective, *if* it beats the logit-KL and CKA twins.
  2. S4.5 confusion-pair protection, *if* it beats CATRO on per-class or confusion-pair accuracy.

  No paper found here conditions functional-graph preservation on correctness. A claim needs an empirical
  win over those twins, not the framing alone.
- **Practical implication:** add the CKA twin, the adaptive-BN-accuracy predictor, CHIP and Wang 2026 to the
  plan's baseline list (P11) before Stage 2/4 results are interpreted.

## Sources (checked in this pass)
- CKA layer pruning: https://arxiv.org/abs/2405.17081 · alternating CKA pruning: https://arxiv.org/abs/2506.04513 · MPruner: https://arxiv.org/abs/2408.13482
- EagleEye: https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3948_ECCV_2020_paper.php
- CHIP: https://proceedings.neurips.cc/paper/2021/hash/ce6babd060aa46c61a5777902cca78af-Abstract.html
- Laenen: https://proceedings.mlr.press/v221/laenen23a.html
- Functional connectomes: https://ojs.aaai.org/index.php/AAAI/article/view/34265
- ACSP: https://arxiv.org/abs/2505.13225 · SIRFP: https://arxiv.org/abs/2412.12672
- NISP: https://arxiv.org/abs/1711.05908
- Neuron Merging: https://arxiv.org/abs/2010.13160
- Central Filter: https://arxiv.org/abs/2112.05493
- KD-guided: https://arxiv.org/abs/2408.11796 · https://arxiv.org/abs/2307.00198
- Class-aware: https://arxiv.org/abs/2312.05875 · CATRO: https://arxiv.org/abs/2110.10921
- OBC: https://arxiv.org/abs/2208.11580
- Spectral structural distortion: https://arxiv.org/abs/2605.18860
- ASF-S (arXiv): https://arxiv.org/abs/2609.24401
