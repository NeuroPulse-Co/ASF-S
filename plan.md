# NeuroPulse — Research Record Standard + Showcase Site

## Context

The NeuroPulse-Co org (FYP: structure-function-aware pruning for continual learning) has three repos:

- **ASF-S** (public). Five lines of work spread over unmerged branches: `cifar_100` (port of the authors' code and an audit), `cifar_10_Conv2Net` (from-scratch reproduction, steps 0–8 checked against "gates"), `mnist_LeNet`, `Bert`, and `afc_prune` (afcprune S0–S5, the foundation for the TIL framework). The results are real and well logged: three LOG.md files in the same format, plus `gate.json`, `results.csv` and PNG figures. But they are scattered and hard to follow, and the top-level README is stale.
- **ASF-S_Hard_Freezing**. Contains only a README. It will be made public.
- **NeuroPulse-Co.github.io**. A hand-written HTML site (Home / Proposal (Canva) / Timeline Gantt / Team) with a dark teal theme. The header is copied into every page, and there are some dead links.

**Goal:** make the GitHub presence a continuously updated, blog-style showcase of the research for three audiences: evaluators, the public and researchers. It should use a friendly first-person researcher voice ("we"), full sentences, defined terms and diagrams, and report negative results openly.

**Precondition (from you):** before anything is published, every repo needs **one unified way to write logs and results**. The site only renders what that standard produces, so anyone (or any new repo) starting mid-project only has to follow the standard.

## ✅ P1 built locally 2026-11-25 (awaiting push by user). Detail kept below for reference.

I build the files locally as two sibling folders next to ASF-S, each with `git init` and nothing committed:
- `/home/dewmina-wijekoon/UoM/FYP_NeuroPulse/NeuroPulse-Co/.github/`
- `/home/dewmina-wijekoon/UoM/FYP_NeuroPulse/NeuroPulse-Co/research-template/`

You create the GitHub repos and push. The `.github` repo must be **public**, because profile READMEs and reusable workflows require it. Commands:
```
gh repo create NeuroPulse-Co/.github --public --source .github --push
gh repo create NeuroPulse-Co/research-template --public --source research-template --push && gh repo edit NeuroPulse-Co/research-template --template
```

### Constraints this design respects
- **The GPU server has no commit rights.** It can `git pull` and run `git rev-parse`, and it writes outputs outside the repo (`~/asfs_outputs`). The laptop rsyncs the results back and you commit. So run records are **written on the server into the outputs root**, and then **harvested** into the repo on the laptop.
- **PyYAML is not in the system Python.** `record.py` carries a PEP 723 header (`dependencies = ["pyyaml"]`) so that `uv run tools/record.py …` works with no setup. It also runs under any Python that has PyYAML, such as the existing venvs. Without PyYAML it exits with a one-line fix.
- **Old LOG entries stay valid.** The new LOG header line is required only for entries dated on or after the adoption date, 2026-11-25.

### `.github` repo contents
```
profile/README.md            org landing page: one-paragraph story, research pipeline diagram (W0–W5), status table, links (site, repos)
standards/
  PLAN.md                    this plan (moved from ASF-S/plan.md)
  RECORDS.md                 the standard: hierarchy, IDs, folder layout, field reference, status vocabulary, run lifecycle, promotion, integrity rules
  STYLE.md                   writing contract (voice, abbreviations, figure captions, result structure, negative results, citing)
  workstreams.yaml           W0–W5: id, title, one-line question, repo(s), owner   (single source; site + record.py read it)
  people.yaml                name, github, role (student/supervisor), workstreams — owner/author names are validated against it
templates/
  study.yaml  run.yaml  LOG-entry.md  FINDINGS.md  post.md  study-page.md  repo-README.md
tools/
  record.py                  canonical copy (vendored into repos)
  tests/test_record.py       pytest, temp dirs only
.github/
  workflows/validate-records.yml   reusable (workflow_call): setup-python, pip install pyyaml, python tools/record.py check
  workflows/notify-site.yml        reusable: secrets.SITE_DISPATCH_TOKEN → POST /repos/NeuroPulse-Co/NeuroPulse-Co.github.io/dispatches {event_type: showcase-updated}
  workflows/ci.yml                 this repo's own CI: pytest tools/tests + record.py check on templates
  ISSUE_TEMPLATE/study-proposal.yml  form: question, hypothesis, success criterion (pre-registered), workstream, owner
  ISSUE_TEMPLATE/config.yml
  pull_request_template.md         checklist: LOG entry added, run.yaml present, check passes, style rules for showcase text
```

### Schemas (enforced by `record.py check`; the site's zod schemas in P3 mirror them)
- **Common.**
  - Slugs match `^[a-z0-9][a-z0-9-]*$`.
  - Workstream ids and person names must exist in `workstreams.yaml` and `people.yaml`. The vendored copy of these files sits in `tools/standards/` and is refreshed by `self-update`.
- **study.yaml.**
  - Required: `id`, `title`, `workstream`, `status` ∈ {planned, running, closed-positive, closed-negative, closed-inconclusive, abandoned}, `owner[]`, `started`, `question`, `hypothesis`, `success_criterion`, `criterion` ∈ {pre_registered, post_hoc}, `code[]`.
  - When status is `closed-*`, also required: `closed`, `verdict`, `key_results[]`.
  - Each item in `key_results` needs `label`, `value`, `unit`, `split` ∈ {val, test} and `n_seeds`. Optional: `std`, `ci95`, `compare{label,value}`.
- **run.yaml.**
  - Required: `id` (`YYYYMMDD-HHMM-<host>-<tag>`), `study`, `author`, `host`, `started`, `finished`, `exit_code`, `commit`, `dirty`, `command`, `seed` (may be null), `backfilled`, `outputs_dir`.
  - Optional: `gpu`, `python`, `torch`, `config`, `notes`, `remote_outputs`.
  - When `backfilled: true`, `source` is required (for example "LOG.md 2026-11-24 entry; gate.json").
- **LOG.md.**
  - Entries are `# YYYY-MM-DD — headline`, newest first.
  - New entries carry the line `Study: · Author: · Host: · Commit: · Runs:` and the sections `## Run`, `## Findings`, `## Next`.
- **showcase/.**
  - `manifest.yaml` is generated, never hand-edited.
  - `posts/*.md` frontmatter: `title`, `date`, `authors[]`, `workstream`, `summary`; optional `study`, `cover`, `draft`.
  - `studies/<id>.md` frontmatter is a subset of study.yaml plus `repo` and `summary`.
  - Every relative image link in the markdown must resolve to a file.

### `tools/record.py` commands (one file, about 500 lines)
| Command | Where | What |
|---|---|---|
| `new <slug> --workstream W2 --owner "…"` | laptop | scaffold `studies/<slug>/` from templates (study.yaml with status planned, LOG.md, runs/) |
| `run <study> --tag T [--seed N] [--outputs-dir P] -- <cmd…>` | server/any | create `$NP_OUTPUTS/<repo>/<study>/<run-id>/`; write run.yaml (git commit and dirty flag via `git`, host, GPU via nvidia-smi, python/torch versions), export `NP_RUN_DIR` to the command, tee `console.log`, then fill in `finished` and `exit_code`. `--outputs-dir` records where legacy scripts write (for example `$ASFS_OUTPUTS/run01/...`) |
| `harvest [study] --remote user@host --root ~/np_outputs/<repo>` | laptop | rsync run.yaml plus small files (yaml/csv/json/png/svg/md, ≤2 MB each; console.log tail only) into `studies/<study>/runs/<run-id>/`, also pulling from each run's `outputs_dir` |
| `log <study>` | laptop | prepend a pre-filled LOG entry (date, author from `git config`, commit, latest runs) |
| `check [--strict]` | laptop + CI | validate everything above; exit code 1 with readable errors (`file:field: message`) |
| `promote <study>` | laptop | copy figures to `showcase/figures/<id>/`, write `showcase/studies/<id>.md` from study.yaml + FINDINGS.md, write a post stub `showcase/posts/<date>-<id>.md` (draft: true), regenerate `manifest.yaml` |
| `self-update` | laptop | fetch the latest record.py + standards files from the `.github` repo's raw URL; show a diff before replacing |

`NP_OUTPUTS` defaults to `./outputs` (gitignored). On the server it is set to `~/np_outputs`. The existing `ASFS_OUTPUTS` is left unchanged and is referenced through `--outputs-dir`.

### `research-template` repo
`README.md` (from the repo-README template: workstream, study table, how to run, how to record) · `studies/.gitkeep` · `showcase/manifest.yaml` (empty, generated) · `tools/record.py` + `tools/standards/{workstreams,people}.yaml` · `.github/workflows/records.yml` (on push/PR: calls validate-records; on push to main touching `showcase/**`: calls notify-site) · `.gitignore` (outputs/, data/, checkpoints/, *.pt, *.npz, console.log).

### Content decisions to confirm while writing
- **people.yaml mapping.** I can't safely guess which GitHub handles belong to which students (NipunSachintha, ravindusw, Semini-Sawbhagya, dewminawijekoon; the unknown contributors Charuka09 and llckbhk). I'll fill in the names from the website's team page and mark the handles `TODO-confirm` for you to correct.
- **Workstream owners** are left as `TBD` until the team assigns them.

### P1 verification
- `uv run --with pytest pytest .github/tools/tests`. The tests cover: check passes on valid templates; check fails with a precise message for a missing field, a bad status, an unknown owner, a closed study without key_results, and backfilled without source; `run` writes a complete run.yaml for `-- python -c "print(1)"` and records a non-zero exit code; `promote` produces a manifest and pages that pass check.
- Smoke test: copy research-template to a temp folder, then `new demo` → `run demo -- echo hi` → `harvest` (local path) → `log demo` → `promote demo` → `check` passes.
- Workflow YAML is linted with `actionlint` if available (otherwise a YAML parse check). The real CI run happens after you push.

## Reviewer critique (supervisor / PhD reviewer view), checked against the repos on 2026-11-25

### A. Research problems (more important than the website)
1. **Our evidence contradicts the proposal's claims.** The proposal (main.tex §Novelty) promises a "replay-free, fine-tuning-free" framework and argues that magnitude scores (PackNet) are flawed. Our own results show two things:
   - Without retraining, recovery is −10.50 points against the paper's −3.78.
   - With full fine-tuning, every selector lands at about −4.8, so the choice of selector stops mattering.
   - Corrected PGI also loses to L1 magnitude, Taylor and BN-γ.
   A reviewer would ask why the framework is built on a criterion that loses to the one it criticises.
2. **The method is not well defined.** The proposal's PGI (gradient SVD per weight) is not the paper's PGI (AFC-based, per filter), and neither one preserves functional connectivity as written. The central "structure-function aware" claim has no working definition yet.
3. **The schedule and scope do not match.** Phase 3 (Oct 19 – Jan 19) promises PGI with hard freezing on Split-MNIST and Split-CIFAR-10, plus ResNet-18. No continual-learning code exists today, and there are about 16 weeks until the final deliverables. Reproducing the pruning paper (CIFAR-100, CIFAR-10, BERT, afcprune S0–S5) has taken over the project, even though the contribution is a continual-learning framework.
4. **The evaluation design is weak.**
   - Hard-frozen masks in task-incremental learning give zero backward transfer by construction, so the headline forgetting metric says nothing.
   - Approach 1 is PackNet with a different score. Given point A1, it will probably lose to PackNet.
   - Only small models are tested (Conv2Net, Conv6Net).
   - Three seeds with a standard deviation of about 1 point is too few to separate the selectors without confidence intervals or paired tests.
5. **Paper and code permissions.** *Resolved:* the paper is now published, so we can cite it and discuss it openly. *Still open:* `min_scripts/` is the authors' code, given to us privately, and it sits in a public repo. We need the author's explicit OK to redistribute it. Without that OK, it moves out of the public repo (and out of its history, using `git filter-repo`). Etiquette: send the reproduction findings to the authors before the posts about them go live.
6. **Individual contributions are uneven.** Git authors: dewminawijekoon 40 commits, ravindusw 5, Charuka 3, llckbhk 2, and two listed team members with 0. FYP evaluators assess individual contribution, and a public showcase makes this visible. Non-code work (literature review, proposal) is not recorded anywhere.
7. **Integrity of the records.** Back-filled run records and success criteria written after the fact must be labelled as such (`backfilled: true`, `criterion: post_hoc`). Otherwise they look like real provenance or real pre-registration. Results must also say whether they are validation or test numbers. afcprune is careful about this; cifar10 step 8 reports test accuracy.

### B. Problems with the plan itself
1. **Too much infrastructure for the time left.** Six or more repos, reusable workflows, dispatch tokens, record.py, a template repo, an Astro rebuild, a restructure and a subtree split come to roughly 2–3 person-weeks, landing in the critical continual-learning phase.
2. **Splitting afcprune into its own repo creates a cross-repo dependency**, because the continual-learning code needs its scorers. It also risks breaking the S1 run that is still going.
3. **The site will show empty workstreams in public.** W3 and W4 would sit empty for months.
4. **Posts should do double duty.** They should be written as drafts of thesis chapters, so the showcase work is not extra work.

## Research plan (addresses A1–A4, A6)

**Deliverable for the supervisors:** a 2-page direction memo containing the findings so far, the go/no-go result, the reframed contributions, and the revised timeline. It is published on the site as the first post of W2.

### R0. Go/no-go and method fix (Nov 26 – Dec 6)
- **One written definition of PGI** (`docs/method/pgi.md`). It reconciles the paper's AFC-based per-filter PGI with the proposal's gradient-SVD per-weight PGI and picks one for continual learning. Structured, per-filter masks are recommended. The definition comes with a unit test.
- **Make "structure-function aware" testable.** Use the existing fidelity module in `afcprune` to measure how well each selector preserves the functional-connectivity matrix. If PGI does not preserve it better than L1 or Taylor, that claim is dropped.
- **Finish S1 and run S2** ("does AFC add information beyond logit matching?"). Freeze afcprune after S2. Stages S3–S5 amount to a separate pruning paper and are out of scope.
- **Go/no-go criterion, written down before running:** on single-task pruning, either corrected PGI comes within the 95% CI of the best simple selector at matched kept-parameters, or it preserves functional connectivity measurably better. If neither holds, PGI becomes one scorer among several, and the contribution shifts to C2 and C3 below.

### Reframed contributions (each one survives whichever way R0 goes)
- **C1. Function-aware selection for parameter isolation.** Kept only if R0 passes.
- **C2. Adaptive per-task capacity** using the cumulative-variance threshold τ. This works with any scorer, and it targets the real weakness of PackNet's fixed ratio. It is measured by capacity used per task at equal accuracy.
- **C3. A careful reproduction study of ASF-S.** This is mostly done already (CIFAR-100 audit, CIFAR-10 steps 0–8, S1). It becomes a thesis chapter and a candidate submission to a reproducibility track.
- **"Zero retraining" becomes "measured retraining overhead".** Retraining epochs and FLOPs per task are reported, never assumed to be zero.

### R1. Continual-learning evaluation protocol (factorial, so every cell is informative)
- **Design:** scorer {PGI, L1, Taylor, random} × allocation {fixed p%, adaptive τ} × protection {hard freeze (W3), metaplastic soft (W4)}.
- **Benchmarks:** Split-MNIST as a sanity check, Split-CIFAR-10 (5 tasks of 2 classes), Split-CIFAR-100 (10 tasks of 10 classes). Models: Conv6Net, and ResNet-18 only if time allows.
- **Baselines:** fine-tuning (lower bound), separate networks (upper bound), PackNet, WSN and EWC. Reuse existing implementations where possible: check Avalanche or Mammoth before writing our own.
- **Metrics:** average accuracy, capacity used per task, retraining overhead and forward transfer. Backward transfer is reported but flagged as 0 by construction for hard freezing, so it is meaningful only for W4.
- **Statistics:** 5 seeds, mean ± 95% CI, paired tests against PackNet. Tuning uses validation data only, and the test set is looked at once, reusing afcprune's `protocol` test-access guard.

### Revised timeline (today is 2026-11-25)
| Window | Research | Infra (in parallel, different owner) |
|---|---|---|
| Nov 26 – Dec 6 | R0: PGI definition, functional-connectivity fidelity test, S1/S2, go/no-go; supervisor memo | P1: standard, templates, `.github` repo |
| Dec 7 – Dec 27 | Continual-learning harness, PackNet reproduced, hard freezing with pluggable scorer (W3) on Split-MNIST and Split-CIFAR-10 | P2: restructure + split repos (after S1 finishes) |
| Dec 28 – Jan 24 | Factorial on Split-CIFAR-10, adaptive τ, metaplastic soft consolidation (W4) | P3–P5: site, backfill posts, automation |
| Jan 25 – Feb 9 | Split-CIFAR-100, Conv6Net (ResNet-18 if time) | posts per milestone |
| Feb 10 – Mar 15 | Final runs (single test look), thesis, poster | thesis chapters drafted from posts |

The Timeline page data is updated to match this table.

### Ownership (fixes A6)
Each workstream gets a named owner and each study gets an `owner:` field. Suggested split across the 4 members: records/site + C3 · W3 hard freezing · W4 metaplastic · baselines/evaluation harness. The team assigns the names. Non-code work (literature review, proposal, memo) is recorded as studies or posts with authors, so it shows on the site.

## Decisions already made

- **Data flow:** a showcase contract. Each repo has a curated `showcase/` folder **on main only**, and the site's CI pulls it.
- **Site:** Astro + Markdown, kept simple. No CMS, no database, no heavy chart framework.
- **Openness:** failed reproductions and losing baselines are reported as findings.
- **Posts:** one per closed milestone. Any member can write one, and the author is named.
- **Sources:** the seed paper is published. Cite it and explain it in our own words with our own diagrams (the paper link goes in the glossary and citations). The authors' private code (`min_scripts/`) stays public only with the author's OK.
- **Infrastructure scope:** the full plan, run in parallel with the research by a separate owner.
- **Private repos:** made public before they are shown.

---

## 1. Whole-project architecture

### 1a. Research pipeline (what the project *is*)

```
W0 Foundations ──► W1 Reproduce ASF-S ──► W2 AFC pruning protocol ──► W3 TIL: Hard Freezing ──┐
(lit review,        (cifar100 port+audit,    (afcprune S0–S5:            (ASF-S_Hard_Freezing)    ├─► W5 Benchmark & Thesis
 proposal)           cifar10 from scratch,    selectors, recovery tiers,  W4 TIL: Metaplastic      │   (Split-MNIST/CIFAR-10/100 vs
                     mnist sanity, BERT)      FC fidelity)                Soft Consolidation ──────┘    PackNet, WSN, EWC; report, poster)
```

The hierarchy is **Workstream → Study → Run → Log entry**.

| Level | What it is | Example | Lives in |
|---|---|---|---|
| **Workstream** (W0–W5) | A major research thread. Stable for the whole project. | W2 AFC pruning protocol | site repo `src/content/workstreams/*.yaml` |
| **Study** | One question with a hypothesis and a verdict. Maps to an existing "stage" or "step series". | `asfs/afcprune-s1` | `<repo>/studies/<study>/study.yaml` |
| **Run** | One execution: a command, a commit, a seed and an output folder. | `20261124-server-clean_s42` | `<repo>/studies/<study>/runs/<run>/run.yaml` |
| **Log entry** | A dated lab-notebook entry (what we ran, found, and do next). | `2026-11-24 — S1 closed-form done` | `<repo>/studies/<study>/LOG.md` |

Study status vocabulary (fixed): `planned · running · closed-positive · closed-negative · closed-inconclusive · abandoned`.

### 1b. Information pipeline (how work reaches the site)

```
 work branch (messy)                       main (curated)                     site repo                 GitHub Pages
 ─────────────────                         ──────────────                     ─────────                 ────────────
 studies/<s>/LOG.md       ──study closes──► showcase/                          sync-showcase.mjs        neuropulse-co.github.io
 studies/<s>/runs/*/run.yaml  (promote)       manifest.yaml (studies+status) ─► reads registry.yaml ──► Astro build
 results CSV/JSON/PNG                         studies/<s>.md  (result page)     sparse-clones each      (schema-validated;
 study.yaml                                   posts/*.md      (blog post)       repo's showcase/        bad manifest = failed build)
                                              figures/*.png|svg
                    research repo push to main touching showcase/ ──repository_dispatch──► rebuild (also nightly + manual)
```

## 2. Unified record standard (do this first)

This lives in a new org repo **`NeuroPulse-Co/.github`**. GitHub uses that repo for the org profile and default community files. It contains:

```
.github/
  profile/README.md                 org landing page on github.com/NeuroPulse-Co (story, workstream status, site link)
  standards/RECORDS.md              the standard: hierarchy, IDs, status vocab, folder layout, promotion rule
  standards/STYLE.md                writing guide (below)
  templates/study.yaml              ┐
  templates/run.yaml                │ copy-paste templates
  templates/LOG-entry.md            │
  templates/post.md                 │
  templates/study-page.md           ┘
  tools/record.py                   one-file helper, vendored into each repo (see below)
  .github/workflows/
    validate-records.yml            reusable workflow: checks study.yaml / run.yaml / showcase manifest against schema
    notify-site.yml                 reusable workflow: repository_dispatch to the site on showcase/ changes
  ISSUE_TEMPLATE/study-proposal.md  open a Study as an issue: question, hypothesis, success criterion
  pull_request_template.md
```

Also create a **template repo `NeuroPulse-Co/research-template`**: the empty skeleton below, with `tools/record.py` and the two workflow callers already wired. Every new repo is created from it, which is how work that "starts from the middle" stays conformant.

### Per-repo layout

```
<repo>/
  README.md            what the repo is, which workstream, study table (links), how to run
  studies/
    <study-slug>/
      study.yaml       question, hypothesis, success criterion, status, owner, dates, workstream, paper targets
      LOG.md           dated entries, newest first (existing format kept)
      runs/<run-id>/   run.yaml + small outputs (csv/json/png); big artifacts stay on server (~/asfs_outputs)
      FINDINGS.md      written when the study closes: verdict + evidence + what it means
      <code…>          code may stay where it is; study.yaml points to it via `code:` paths
  showcase/            (main only) curated, public-facing copy — see §3
  tools/record.py
```

Each study folder is self-contained: its code, record files and small results all live in `studies/<slug>/`. Existing work is physically moved into this layout (see P2), not just pointed at.

### Templates (key fields)

`study.yaml`
```yaml
id: afcprune-s1                 # unique within repo; global id = <repo>/<id>
title: Does AFC-preserving selection beat simple filter scores?
workstream: W2
status: closed-negative
owner: [Dewmina Wijekoon]
started: 2026-11-10
closed: 2026-11-24
question: >                     # full sentences, no bare abbreviations
hypothesis: >
success_criterion: >            # decided BEFORE running
code: [afcprune/afcprune, afcprune/experiments/run_s1.sh]
paper_targets: {conv2net_t5_dacc: -3.78}
verdict: >                      # filled at close
key_results:                    # 1–5 headline numbers the site shows as a scorecard
  - {label: "PGI (corrected), 75.8% params removed", value: -8.4, unit: "Δ acc pts", compare: {label: paper, value: -3.78}}
```

`run.yaml` (written by `tools/record.py`, not by hand)
```yaml
id: 20261124-server-clean_s42
study: afcprune-s1
date: 2026-11-24T13:20:00+05:30
author: Dewmina Wijekoon
host: server   gpu: RTX…   commit: 20147c2   dirty: false
command: bash experiments/run_s1.sh clean_s42
seed: 42
config: {…}                     # dumped config dict
outputs: [results.csv, gate.json, alloc.json]
remote_outputs: ~/afcp_outputs/s1/clean_s42   # large files not in git
```

`LOG.md` entry. This is the existing format plus a short header line, so old entries remain valid:
```markdown
# 2026-11-24 — S1 closed-form tiers finished; PGI loses to Taylor/BN-γ/L1
Study: afcprune-s1 · Author: Dewmina · Host: server · Commit: 20147c2 · Runs: clean_s42..s44
## Run      (command / runs / tests table)
## Findings
## Next     (checklist)
```

`tools/record.py` (one file, standard library + PyYAML only):
- `record.py start <study> --tag … -- <cmd>`: creates `runs/<id>/` and writes `run.yaml` (git commit, dirty flag, host, GPU, seed, command, time), runs the command and copies the declared small outputs.
- `record.py log <study>`: appends a pre-filled LOG entry stub.
- `record.py check`: validates all study.yaml and run.yaml files. The same check runs in CI.
- `record.py promote <study>`: copies study.yaml, FINDINGS, the selected figures and a stub post into `showcase/` for the PR to main.

This generalises what already exists (`scripts/archive_run.sh` → RUN_INFO.md, and `gate.json`).

### STYLE.md (writing contract for everything that reaches the site)
- First-person plural researcher voice, friendly, complete sentences. No note-style fragments.
- Spell out every abbreviation on its first use on each page and link it to the Glossary (ASF-S, AFC, PGI, PG1, TIL, BWT, FWT, ECE…).
- Every figure gets a caption that says *what to look at* and *what it means*.
- Every result states: the question, what we expected, what happened, and why we think it happened, plus what we do next.
- Negative results are stated plainly, with the evidence.
- The seed paper is cited, never quoted at length. Method diagrams are our own.

## 3. Showcase contract (`showcase/` on main)

```
showcase/
  manifest.yaml        repo: ASF-S; workstreams: [W1, W2]; studies: [list of study ids + status + summary]
  studies/<id>.md      result page: frontmatter = study.yaml subset; body = FINDINGS in blog voice + figures + tables
  posts/<date>-<slug>.md   milestone blog post (frontmatter: title, date, authors, study, workstream, summary, cover)
  figures/<id>/*.png|svg
  data/<id>/*.csv      optional small tables rendered as HTML tables
```

The site validates all frontmatter with Astro content-collection schemas (zod). Any file that violates the contract fails the build, so the template is enforced mechanically.

## 4. Site architecture (rebuild NeuroPulse-Co.github.io in Astro, simple)

Port the existing theme from `assets/css/base.css` (colour tokens, Inter + Space Grotesk, `.card`, `.badge`, `.callout`) into one layout. Reuse `assets/js/timeline.js` and `timeline-data.js` as the Timeline page, now reading from a data file.

```
site/
  registry.yaml                     list of research repos to pull (add a repo = add a line)
  scripts/sync-showcase.mjs         sparse-clone each repo's showcase/ → src/content/{studies,posts}/<repo>/…, figures → public/
  src/content.config.ts             zod schemas: workstreams, studies, posts, glossary, team
  src/content/workstreams/W0..W5.yaml   (site-owned: title, plain-language summary, diagram)
  src/content/glossary.yaml
  src/content/team.yaml
  src/layouts/Base.astro            shared header/footer (fixes the copy-paste duplication)
  src/pages/
    index.astro          story hero + "where we are now" status board (workstreams × studies, coloured by status) + latest posts
    research/index.astro           the research pipeline diagram (§1a) with clickable workstreams
    research/[workstream].astro    workstream page: plain-language explainer + its studies
    studies/[...id].astro          study result page with scorecard (ours vs paper), figures, link to code/runs
    blog/index.astro, blog/[...slug].astro   milestone posts, filter by workstream/author
    timeline.astro                 Gantt (existing code), phases + study dates
    glossary.astro, team.astro, proposal.astro (Canva embed + our summary)
  .github/workflows/deploy.yml      on: push, repository_dispatch, nightly cron, manual → sync → astro build → Pages
```

- Diagrams: hand-drawn SVG (Excalidraw or draw.io export) stored in `figures/`, plus Mermaid rendered on the client for simple flowcharts. No server-side rendering toolchain.
- Charts: committed PNGs from the research code are the default. A CSV table component is the only "live" data rendering. Interactive charts can wait.
- Pages switches from `legacy` (branch) to a GitHub Actions build.

## 5. Execution order

**Step 0 (done 2026-11-25).** Copied this plan to `ASF-S/plan.md` so that other agents and team members can refer to it. It moves to `.github/standards/PLAN.md` in P1, and it is re-copied after each plan update.

**P1: Standard (prerequisite).** Create the `.github` repo with RECORDS.md, STYLE.md, templates, `tools/record.py` and the two reusable workflows. Create `research-template`. Write the org profile README.

**P2: Restructure existing work.** Repos are aligned one-to-one with workstreams, and all unmerged branches are folded into main so that nothing lives only on a side branch.

Target repo map:

| Repo | Workstream | Contents after restructure |
|---|---|---|
| `ASF-S` (kept, renamed in README to "ASF-S Reproduction") | W1 | studies below; `reference/min_scripts` (authors' code, the spec) |
| `AFC-Prune` (**new**, extracted from `ASF-S/afcprune/` with `git subtree split` so history is kept; its README already says it "will move to its own repo") | W2 | studies `s0-skeleton`, `s1-baselines`, later `s2`…`s5`; the `afcprune` package |
| `ASF-S_Hard_Freezing` (made public, created from the template) | W3 | empty study skeleton |
| `Metaplastic-Soft` (later, from the template) | W4 | — |
| `.github`, `research-template`, `NeuroPulse-Co.github.io` | infra | — |

ASF-S study mapping (source branch → new folder):

| New folder | Source | Status |
|---|---|---|
| `studies/cifar100-port-audit/` | `cifar_100`: cifar100/, scripts/, docs/cifar100-status.md, docs/asf-s-paper-vs-implementation.md, runs/*/RUN_INFO.md, plus main's docs/engineering-log.md | closed-negative (bugs, PG1 collapse) |
| `studies/mnist-lenet-sanity/` | `origin/mnist_LeNet`: mnist/ | to confirm |
| `studies/bert-pg-transfer/` | main: Bert/ | to confirm |
| `studies/cifar10-repro/` | `afc_prune`: cifar10/ (steps 0–7 pass, step 8 −10.50 vs −3.78) | closed-negative |
| `archive/kaggle/` | notebooks/kaggle_conv2net.ipynb | abandoned |
| `reference/min_scripts/` | authors' code | public only with the author's written OK; otherwise removed from the repo and its history |

Record-integrity rules for the retrofit: back-filled `run.yaml` files carry `backfilled: true`. Studies whose success criterion was written after the results carry `criterion: post_hoc`. Every reported number is tagged `split: val|test`.

Restructure steps, done on a `restructure` branch for you to review and merge:
1. Merge `cifar_100`, `origin/mnist_LeNet` and `afc_prune` into the branch, resolving conflicts in shared files (README, .gitignore, requirements).
2. `git mv` each line of work into its `studies/<slug>/` folder, keeping history. Fix the path assumptions that break: `run.sh`/`setup.sh`, `config.py` output roots (`ASFS_OUTPUTS` stays the server root), `scripts/run_pipeline.sh`, and test imports. Re-run each study's tests (`cifar10/tests`, `afcprune/tests` with golden.npz) to confirm nothing broke.
3. Extract `afcprune/` into the new `AFC-Prune` repo with `git subtree split`, then remove it from ASF-S with a pointer in the README.
4. In each study: write study.yaml, convert existing run outputs into `runs/<id>/run.yaml` (backfilled from LOG entries, RUN_INFO.md and gate.json), prepend the standard header to existing LOG entries (no rewriting of content), and write FINDINGS.md for closed studies.
5. Replace the stale top-level README with the repo overview and study table. Delete the merged branches after you confirm.
6. Promote each study to `showcase/`.

Server note: output folders on the server (`~/asfs_outputs`, `~/afcp_outputs`) are not moved. Only the code paths change, so the server checkout needs a fresh rsync of the new layout before the next run.

**P3: Site.** Scaffold Astro, port the theme, add schemas, the sync script and the deploy workflow. Write the site-owned content: W0–W5 explainers, glossary, team, our own ASF-S method diagram, and the proposal summary. Fix the known site issues (dead `#/approach-*` links, "six phases" vs 5, "AFS" typo, stale README).

**P4: Backfill posts.** Write one post per closed study, plus an intro post ("What we are building and why"). Since posts are per milestone and written by any member, I can draft these from the LOG and FINDINGS for the owner to edit.

**P5: Automation.** Add `notify-site.yml` and `validate-records.yml` to each research repo, and add the repository_dispatch token secret (you set this up).

## Verification

- `python tools/record.py check` passes on ASF-S. A deliberately broken study.yaml fails it locally and in CI.
- `npm run sync && npm run build` in the site succeeds. A showcase post with a missing `authors` field fails the build with a clear schema error.
- `npm run preview`: check every page at desktop and 375px width, the status board matches the study statuses, figures load, and glossary links resolve.
- Run a link checker (e.g. `lychee`) on the built `dist/`.
- End to end: push a trivial edit to `ASF-S/showcase/` on main → dispatch fires → the deploy action runs → the change is live on neuropulse-co.github.io.

## Notes
- You make all commits and pushes. I prepare files and PR-ready changes only.
- Large artifacts stay on the server (`~/asfs_outputs`, `~/afcp_outputs`). `run.yaml` records where they are.
- Still open, to decide in P2: the status of the mnist and BERT studies (I will read their results and propose one).
- The afcprune S1 gradient tiers are still running. The restructure should land between runs so the server code is not changed mid-run.
- Restructure verification: before and after the move, `pytest` passes in cifar10 and AFC-Prune, and `git log --follow` on a moved file still shows its history.
