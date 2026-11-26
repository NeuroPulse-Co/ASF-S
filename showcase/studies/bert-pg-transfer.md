---
id: bert-pg-transfer
repo: ASF-S
title: Does principal-gradient pruning transfer to the feed-forward layers of BERT?
workstream: W1
status: closed-inconclusive
owner:
- Dewmina Wijekoon
started: '2026-01-06'
closed: '2026-04-28'
summary: We applied the principal-gradient (PG) importance score from Artificial Structure Function Search
  (ASF-S) to the feed-forward neurons of a BERT-base text classifier on AG News. Pruning those neurons
  in three of BERT's twelve layers costs about one point of accuracy or less for every method we tried,
  so on one seed the results cannot tell PG apart from random or magnitude pruning.
question: When we prune feed-forward (FFN) intermediate neurons in layers 0, 5 and 11 of a BERT-base model
  fine-tuned on AG News, does ranking neurons by PG importance keep more test accuracy and better calibration
  than random, L1 or L2 magnitude pruning?
hypothesis: If the first principal gradient (PG1) of the functional-connectivity graph between FFN neurons
  marks the neurons that separate correct from incorrect predictions, as it is meant to in convolutional
  networks, PG-ranked pruning should lose less accuracy than random or magnitude pruning at the same share
  of neurons removed.
success_criterion: At matched shares of FFN neurons removed, PG pruning keeps clearly more test accuracy
  than random, L1 and L2 pruning, by more than the measurement noise of the AG News test set (about 0.26
  points at 94.5% accuracy on 7,600 examples).
criterion: post_hoc
verdict: The evidence cannot separate PG pruning from the simple baselines. In the committed sweep, zeroing
  99.96% of the FFN neurons in layers 0, 5 and 11 of the fully fine-tuned model lowers test accuracy only
  from 94.54% to 93.57% before retraining, and the later all-method figure keeps PG, random, L1 and L2
  within about one point of each other at every budget, on one seed. Three FFN layers of BERT are too
  redundant, behind the residual connections, for this setup to test whether PG chooses neurons well.
key_results:
- label: BERT-base fully fine-tuned on AG News, unpruned
  value: 94.54
  unit: accuracy (%)
  split: test
  n_seeds: 1
- label: Same model, 99.96% of FFN neurons in layers 0, 5 and 11 zeroed by PG importance, before retraining
  value: 93.57
  unit: accuracy (%)
  split: test
  n_seeds: 1
  compare:
    label: unpruned
    value: 94.54
- label: Frozen-encoder model, 99.51% of FFN neurons in layers 0, 5 and 11 zeroed by PG importance, before
    retraining
  value: 66.59
  unit: accuracy (%)
  split: test
  n_seeds: 1
  compare:
    label: unpruned frozen-encoder model
    value: 83.71
evidence:
- type: diagram
  title: How we moved PG pruning from images to a language model
  caption: We fine-tuned BERT-base on the AG News topic task in two ways, once with the encoder frozen
    and once with every layer trained, and recorded the feed-forward (FFN) neuron activations in layers
    0, 5 and 11. The blue step is the principal-gradient (PG) score, which compares how those neurons
    behave on correctly and incorrectly classified examples. We removed the lowest-scoring neurons, retrained
    and measured test accuracy. The later four-method comparison against random and weight-magnitude choices
    retrained only the classifier.
  src: ../figures/bert-pg-transfer/evidence-01-how-we-moved-pg-pruning-from-images-to-a-languag.svg
  source_url: https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/bert-pg-transfer/diagrams/method.excalidraw.svg
- type: table
  title: Every setting in the committed PG sweep
  caption: PG pruning of BERT's feed-forward neurons, one seed; accuracies are fractions of the AG News
    test set. Even the lowest threshold removes more than 96% of the neurons in the three layers. In the
    fully fine-tuned model, accuracy after retraining stays within about 0.3 points of the unpruned 0.945
    at every threshold, so this sweep cannot tell PG apart from any other way of choosing neurons.
  split: test
  columns:
  - key: model_mode
    label: Starting model
  - key: threshold
    label: PG threshold
  - key: pruned_pct_overall
    label: Neurons removed in layers 0, 5, 11 (%)
  - key: baseline_acc
    label: Unpruned accuracy
  - key: postprune_acc
    label: Accuracy after pruning
  - key: best_retrain_acc
    label: Accuracy after retraining
  labels:
    head: Head-only (encoder frozen)
    full: Fully fine-tuned
  rows:
  - model_mode: head
    threshold: 0.1
    pruned_pct_overall: 96.72309027777779
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.8372368421052632
    best_retrain_acc: 0.8606578947368421
  - model_mode: head
    threshold: 0.2
    pruned_pct_overall: 98.90407986111111
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.825
    best_retrain_acc: 0.8530263157894736
  - model_mode: head
    threshold: 0.3
    pruned_pct_overall: 99.51171875
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.665921052631579
    best_retrain_acc: 0.8501315789473685
  - model_mode: head
    threshold: 0.4
    pruned_pct_overall: 99.65277777777779
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.6655263157894736
    best_retrain_acc: 0.8531578947368421
  - model_mode: head
    threshold: 0.5
    pruned_pct_overall: 99.76128472222221
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.67
    best_retrain_acc: 0.8435526315789473
  - model_mode: head
    threshold: 0.6
    pruned_pct_overall: 99.83723958333334
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.6778947368421052
    best_retrain_acc: 0.8563157894736843
  - model_mode: head
    threshold: 0.7
    pruned_pct_overall: 99.91319444444444
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.6781578947368421
    best_retrain_acc: 0.8452631578947368
  - model_mode: head
    threshold: 0.8
    pruned_pct_overall: 99.92404513888889
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.6784210526315789
    best_retrain_acc: 0.8506578947368421
  - model_mode: head
    threshold: 0.9
    pruned_pct_overall: 99.93489583333334
    baseline_acc: 0.8371052631578947
    postprune_acc: 0.6794736842105263
    best_retrain_acc: 0.8463157894736842
  - model_mode: full
    threshold: 0.1
    pruned_pct_overall: 96.12630208333334
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9393421052631579
    best_retrain_acc: 0.9451315789473684
  - model_mode: full
    threshold: 0.2
    pruned_pct_overall: 98.49175347222221
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9365789473684211
    best_retrain_acc: 0.9422368421052632
  - model_mode: full
    threshold: 0.3
    pruned_pct_overall: 99.62022569444444
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9367105263157894
    best_retrain_acc: 0.9446052631578947
  - model_mode: full
    threshold: 0.4
    pruned_pct_overall: 99.88064236111111
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9355263157894737
    best_retrain_acc: 0.9444736842105264
  - model_mode: full
    threshold: 0.5
    pruned_pct_overall: 99.92404513888889
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9356578947368421
    best_retrain_acc: 0.9465789473684211
  - model_mode: full
    threshold: 0.6
    pruned_pct_overall: 99.94574652777779
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9356578947368421
    best_retrain_acc: 0.9431578947368421
  - model_mode: full
    threshold: 0.7
    pruned_pct_overall: 99.95659722222221
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9356578947368421
    best_retrain_acc: 0.945
  - model_mode: full
    threshold: 0.8
    pruned_pct_overall: 99.95659722222221
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9356578947368421
    best_retrain_acc: 0.9453947368421053
  - model_mode: full
    threshold: 0.9
    pruned_pct_overall: 99.95659722222221
    baseline_acc: 0.9453947368421053
    postprune_acc: 0.9356578947368421
    best_retrain_acc: 0.9447368421052632
  data: ../data/bert-pg-transfer/evidence-02-every-setting-in-the-committed-pg-sweep.csv
  source_url: https://github.com/NeuroPulse-Co/ASF-S/blob/main/studies/bert-pg-transfer/sweep_results.csv
code_url: https://github.com/NeuroPulse-Co/ASF-S/tree/main/studies/bert-pg-transfer
---

<!-- Generated by record.py promote from studies/bert-pg-transfer/study.yaml and FINDINGS.md. Edit those files, not this one. -->

## Verdict

We asked whether the principal-gradient (PG) importance score from Artificial Structure Function
Search (ASF-S), which was designed for convolutional networks, also picks good neurons to prune in
a transformer. We fine-tuned BERT-base on the AG News news-topic task (four classes) and pruned
feed-forward (FFN) intermediate neurons in layers 0, 5 and 11 by PG importance, random choice, and
L1 or L2 weight magnitude. The result is inconclusive. Removing almost all FFN neurons in those
three layers costs about one point of test accuracy or less, whichever method chooses them, so on
a single seed the experiment cannot show that PG chooses better than the baselines. We neither
confirm nor reject the hypothesis.

## Evidence

The pipeline, in `main.py`, fine-tunes BERT-base, records each FFN neuron's activation at the
classification token for correctly and incorrectly classified test examples, builds the
first principal gradient (PG1) of the neuron-to-neuron similarity graph for each class, and turns
the difference between the correct and incorrect PG1 into one importance score per neuron.
Pruning sets a neuron's incoming and outgoing weights to zero. It was run in two regimes: "full",
where the whole model was fine-tuned and is retrained for two epochs after pruning, and
"head-only", where the encoder is frozen and only the classifier is trained. All numbers are test
accuracy on the 7,600 AG News test examples, from one seed (42).

The committed sweep (`sweep_results.csv`, 2026-01-06) uses PG importance only. Its thresholds
remove far more neurons than their names suggest: even the lowest threshold, 0.1, removes 96.7%
(head-only) and 96.1% (full) of the FFN neurons in the three layers, because most neurons have
importance scores close to zero. The table below gives three of its rows.

| Regime | Threshold | FFN neurons removed in layers 0, 5, 11 | Unpruned (%) | After pruning (%) | After retraining (%) |
|---|---|---|---|---|---|
| Full | 0.1 | 96.13% | 94.54 | 93.93 | 94.51 |
| Full | 0.9 | 99.96% | 94.54 | 93.57 | 94.47 |
| Head-only | 0.3 | 99.51% | 83.71 | 66.59 | 85.01 |

In the fully fine-tuned model, zeroing nearly every FFN neuron in three of BERT's twelve layers
costs under one point before any retraining (94.54% to 93.57%), and two epochs of retraining bring
accuracy back to within 0.1 points of the unpruned model. In the head-only model the drop before
retraining is large (83.71% to 66.59%), but retraining the classifier lifts it to 85.01%, above the
unpruned 83.71%. That tells us the head-only starting model was itself undertrained, so its
"recovery" mostly measures extra classifier training.

Later sweeps, from 2026-01-09 to 2026-01-22, fixed the thresholds to remove 10% to 90% of neurons
and added random, L1 and L2 pruning. Their CSV files were not committed, but the figures were. The
figure below compares all four methods on the fully fine-tuned model with only the classifier
retrained after pruning.

![Test accuracy against the share of FFN neurons removed in layers 0, 5 and 11, for PG, random, L1 and L2 pruning, before (dashed) and after (solid) retraining the classifier; one seed. Look at the vertical axis: every curve lies between about 93.8% and 94.8%, and the dotted line is the unpruned 94.54%. PG after retraining is often the top curve, but the gaps between methods are a few tenths of a point, about the size of the test-set noise, so this figure cannot rank the methods.](../figures/bert-pg-transfer/accuracy_all_methods_head.png)

The next figure shows PG alone with all layers retrained. Retraining sometimes ends below the
accuracy measured straight after pruning, which is another sign that the differences here are
noise from two short retraining epochs rather than an effect of which neurons were kept.

![PG pruning of the fully fine-tuned model with all layers retrained, one seed. Look at the retrained curve (green): at 10% and 20% of neurons removed it sits below the post-pruning curve (blue), and at 70% it sits above the unpruned line. The whole range is under one point, so these swings are noise, not a trend.](../figures/bert-pg-transfer/accuracy_ece_vs_pruned_full.png)

## Why we think this happened

We think the setup is not sensitive enough to test the selector. BERT has twelve layers, each with
a residual connection that carries information around the FFN block, and we pruned FFN neurons in
only three of them. Removing an FFN block almost entirely leaves the residual path intact, so the
model loses little no matter which neurons go. When every method costs less than one point, a good
selector and a random one look alike. We are fairly confident in this explanation, because it fits
both the committed sweep (99.96% removed, under one point lost) and the later figures. A study that
pruned all twelve layers, or measured the share of all model parameters removed, would show
whether PG separates from the baselines once pruning actually hurts.

We also saw signs that the PG importance scores are concentrated on few neurons. In the notebook,
half of the neurons in layer 5 score below 0.0044 on a 0-to-1 scale, while the top 1% score above
0.26. With only four classes there are at most four PG1 differences per layer to analyse, so the
importance score rests on very little data. We have not checked whether the scores mostly track
weight magnitude, as they did in our LeNet sanity check (`mnist-lenet-sanity`); the notebook
started a plot of PG importance against weight size but recorded no conclusion.

## Limits

- One seed. The AG News test set has 7,600 examples, so the standard error of an accuracy near
  94.5% is about 0.26 points, which is the size of most differences we see.
- Only FFN neurons in layers 0, 5 and 11 were pruned. The "% pruned" figures are shares of those
  three layers' FFN neurons (9,216 of 36,864 in the model), not of the model's parameters.
  Pruning zeroes weights rather than removing them, so the model does not get smaller or faster.
- Test data was used for everything: building PG1, choosing the best retraining epoch, and
  reporting. There is no validation split, so the retrained numbers are optimistic.
- The committed CSV was written by an earlier version of `main.py` (it lacks the `pruning_method`
  column), and its thresholds behave differently from the current code. We read it as PG pruning
  from the README's commands, but this is a reconstruction.
- The CSV files behind the figures are not in the repository. The figure
  `accuracy_ece_vs_pruned_head.png` is titled "PG-guided" but plots the rows of all four methods
  together, so we do not use it. The "head-only" label also means different things in different
  runs: the frozen-encoder model (83.71% unpruned) in the committed CSV, and the fully fine-tuned
  model (94.54% unpruned) with only the classifier retrained in the later figures.
- One dataset and one model. We do not know how the paper's authors would adapt ASF-S to
  transformers, since their paper covers only convolutional and fully connected networks.

## What this means for the project

This study does not change the project's direction. It tells us that a transformer test of PG
pruning needs a harder setting, such as pruning every layer and reporting the share of all
parameters removed, with several seeds and a validation split, before it can say anything about
the selector. For now our main question, whether functional-connectivity scores beat weight
magnitude, is better answered in the convolutional studies of workstream W2. If we return to
BERT, the existing pipeline in `main.py` is a working starting point.

**Acknowledgements.** This study was designed and run by GitHub user `Charuka09` (Charuka Herath). Dewmina Wijekoon maintains this record.
