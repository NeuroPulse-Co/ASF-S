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

![Test accuracy against the share of FFN neurons removed in layers 0, 5 and 11, for PG, random, L1 and L2 pruning, before (dashed) and after (solid) retraining the classifier; one seed. Look at the vertical axis: every curve lies between about 93.8% and 94.8%, and the dotted line is the unpruned 94.54%. PG after retraining is often the top curve, but the gaps between methods are a few tenths of a point, about the size of the test-set noise, so this figure cannot rank the methods.](accuracy_all_methods_head.png)

The next figure shows PG alone with all layers retrained. Retraining sometimes ends below the
accuracy measured straight after pruning, which is another sign that the differences here are
noise from two short retraining epochs rather than an effect of which neurons were kept.

![PG pruning of the fully fine-tuned model with all layers retrained, one seed. Look at the retrained curve (green): at 10% and 20% of neurons removed it sits below the post-pruning curve (blue), and at 70% it sits above the unpruned line. The whole range is under one point, so these swings are noise, not a trend.](accuracy_ece_vs_pruned_full.png)

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
