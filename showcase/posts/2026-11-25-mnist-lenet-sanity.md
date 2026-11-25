---
title: Does ASF-S pruning on LeNet-300-100 with MNIST reproduce the paper's claims and beat weight magnitude?
date: '2026-11-25'
authors:
- Ravindu Wanigasooriya
workstream: W1
study: mnist-lenet-sanity
summary: We re-implemented the Artificial Structure Function Search (ASF-S) pipeline for the paper's smallest
  experiment, a LeNet-300-100 network on MNIST, as a quick sanity check before the larger reproductions.
  The paper's baseline network reproduces exactly, but ASF-S never beats simple weight-magnitude pruning,
  and the paper's "no accuracy drop at 70% sparsity" claim holds only when every layer is fine-tuned.
draft: true
---

<!-- Draft post written from the study record. Follow standards/STYLE.md, remove every TODO,
     then set draft: false. Posts double as thesis-chapter drafts. -->

## The question

TODO: what we wanted to know, and why it matters for the project.

## What we expected

TODO: our hypothesis and the result that would have confirmed it.

## What we did

TODO: the setup in plain words; link the code and the runs.

## What happened

TODO: the results, with a figure and a caption that says what to look at.

## Why we think it happened

TODO: our explanation, and how confident we are in it.

## What comes next

TODO: the decision this result leads to.
