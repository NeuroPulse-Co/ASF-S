#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["pyyaml"]
# ///
"""record.py: the NeuroPulse research-record tool.

One file and vendored into every research repo as tools/record.py. The canonical copy
lives in NeuroPulse-Co/.github; update a vendored copy with `record.py self-update`.
The standard it enforces is described in NeuroPulse-Co/.github/standards/RECORDS.md.

  record.py new <study> --workstream W2 --owner "Name" [--title "..."]
  record.py run <study> --tag T [--seed N] [--outputs-dir P] [--note "..."] -- <command...>
  record.py harvest [<study>] [--from DIR | --from user@host:/abs/path]
  record.py log <study>
  record.py check [--strict]
  record.py promote <study>
  record.py self-update [--yes]

Run with `uv run tools/record.py ...` (installs PyYAML on the fly) or with any Python
that has PyYAML installed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import getpass
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("record.py needs PyYAML. Run it as `uv run tools/record.py ...` "
             "or install it with `pip install pyyaml`.")

VERSION = "1.0.0"
ORG = "NeuroPulse-Co"
RAW_BASE = f"https://raw.githubusercontent.com/{ORG}/.github/main"
ADOPTION_DATE = dt.date(2026, 11, 25)   # LOG entries from this date on must use the new header

STATUSES = ["planned", "running", "closed-positive", "closed-negative",
            "closed-inconclusive", "abandoned"]
CLOSED = {"closed-positive", "closed-negative", "closed-inconclusive"}
CRITERIA = ["pre_registered", "post_hoc"]
SPLITS = ["val", "test", "none"]   # none: not measured on data (parameter counts, test suites)
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RUN_ID_RE = re.compile(r"^\d{8}-\d{4}-[A-Za-z0-9-]+-[A-Za-z0-9_.-]+$")
LOG_HEAD_RE = re.compile(r"^# (\d{4}-\d{2}-\d{2})(?: (\d{2}:\d{2}))? — (.+)$")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
PLACEHOLDER_RE = re.compile(r"TODO|<headline|<one sentence")
HARVEST_EXT = ["*.yaml", "*.yml", "*.csv", "*.json", "*.jsonl", "*.png", "*.svg", "*.md",
               "*.txt", "console.log"]
HARVEST_MAX = "2m"
CONSOLE_TAIL_LINES = 200

STUDY_REQUIRED = ["id", "title", "workstream", "status", "owner", "started", "summary",
                  "question", "hypothesis", "success_criterion", "criterion", "code"]
RUN_REQUIRED = ["id", "study", "author", "host", "started", "finished", "exit_code",
                "commit", "dirty", "command", "seed", "backfilled", "outputs_dir"]
POST_REQUIRED = ["title", "date", "authors", "workstream", "summary"]
PAGE_REQUIRED = ["id", "repo", "title", "workstream", "status", "owner", "summary"]


# --------------------------------------------------------------------------- helpers

def die(msg: str, code: int = 2):
    print(f"record.py: {msg}", file=sys.stderr)
    sys.exit(code)


def sh(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def find_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    if os.environ.get("NP_ROOT"):
        return Path(os.environ["NP_ROOT"]).resolve()
    top = sh(["git", "rev-parse", "--show-toplevel"])
    return Path(top) if top else Path.cwd()


def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=100)


def split_frontmatter(text: str):
    """Return (frontmatter dict or None, body)."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---", 4)
    if end < 0:
        return None, text
    fm = yaml.safe_load(text[4:end]) or {}
    body = text[end + 4:].lstrip("\n")
    return fm, body


def write_frontmatter(path: Path, fm: dict, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, width=100)
    path.write_text(f"---\n{head}---\n\n{body.rstrip()}\n", encoding="utf-8")


def as_date(v):
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    if isinstance(v, str):
        try:
            return dt.date.fromisoformat(v[:10])
        except ValueError:
            return None
    return None


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone().replace(microsecond=0)


class Ctx:
    """Repo context: root, repo name, standards."""

    def __init__(self, root: Path):
        self.root = root
        self.repo = os.environ.get("NP_REPO") or root.name
        self.studies = root / "studies"
        self.showcase = root / "showcase"
        self.workstreams, self.people = self._load_standards()

    def _load_standards(self):
        here = Path(__file__).resolve().parent
        for d in (self.root / "tools" / "standards", self.root / "standards", here / "standards"):
            ws, pp = d / "workstreams.yaml", d / "people.yaml"
            if ws.exists() and pp.exists():
                return load_yaml(ws) or [], load_yaml(pp) or []
        return None, None

    def ws_ids(self):
        return None if self.workstreams is None else [w["id"] for w in self.workstreams]

    def person(self, name: str | None):
        """Canonical short name for name/short/github, or None if unknown."""
        if self.people is None or not name:
            return name if self.people is None else None
        key = str(name).strip().lower()
        for p in self.people:
            for k in ("name", "short", "github"):
                if p.get(k) and str(p[k]).strip().lower() == key:
                    return p.get("short") or p["name"]
        return None

    def study_dir(self, study: str) -> Path:
        return self.studies / study

    def require_study(self, study: str) -> dict:
        f = self.study_dir(study) / "study.yaml"
        if not f.exists():
            die(f"no study '{study}' (expected {f.relative_to(self.root)}). "
                f"Create it with `record.py new {study} ...`.")
        return load_yaml(f) or {}


def git_info(root: Path):
    commit = sh(["git", "rev-parse", "--short", "HEAD"], cwd=root) or "unknown"
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = sh(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    return commit, branch, (bool(status) if status is not None else None)


def default_author(ctx: Ctx, explicit: str | None) -> str:
    for cand in (explicit, os.environ.get("NP_AUTHOR"),
                 sh(["git", "config", "user.name"], cwd=ctx.root), getpass.getuser()):
        if cand:
            canon = ctx.person(cand)
            if canon:
                return canon
    die("could not match you to standards/people.yaml. Pass --author \"Your Name\" "
        "or set NP_AUTHOR (on the server: export NP_AUTHOR=\"Your Name\").")


# --------------------------------------------------------------------------- new

STUDY_TEMPLATE = """\
# Study record. Field reference: NeuroPulse-Co/.github/standards/RECORDS.md
# Write in full sentences and spell out abbreviations (see STYLE.md).
id: {id}
title: "{title}"
workstream: {ws}
status: planned            # planned | running | closed-positive | closed-negative | closed-inconclusive | abandoned
owner: [{owner}]
started: {today}
summary: >-
  TODO: one or two plain-language sentences a newcomer can understand.
question: >-
  TODO: the single question this study answers.
hypothesis: >-
  TODO: what we expect to happen, and why.
success_criterion: >-
  TODO: the measurable result that would make the hypothesis true. Write it before running.
criterion: pre_registered  # pre_registered if written before results exist, otherwise post_hoc
code: []                   # paths (relative to the repo root) of the code this study uses
# Filled when the study closes:
# closed: YYYY-MM-DD
# verdict: >-
#   What we concluded, in two or three sentences.
# key_results:
#   - {{label: "...", value: 0.0, unit: "accuracy points", split: val, n_seeds: 3, std: 0.0,
#       compare: {{label: "paper", value: 0.0}}}}
"""

LOG_PREAMBLE = """\
<!-- Lab log for study `{id}`. Newest entry first.
     Add an entry with `record.py log {id}`; the format is in standards/RECORDS.md. -->
"""


def cmd_new(ctx: Ctx, a):
    if not SLUG_RE.match(a.study):
        die(f"study id '{a.study}' must be lowercase letters, digits and hyphens")
    if ctx.ws_ids() is not None and a.workstream not in ctx.ws_ids():
        die(f"unknown workstream '{a.workstream}'; known: {', '.join(ctx.ws_ids())}")
    owner = ctx.person(a.owner) if ctx.people is not None else a.owner
    if not owner:
        die(f"unknown owner '{a.owner}'; add them to standards/people.yaml first")
    d = ctx.study_dir(a.study)
    if d.exists():
        die(f"{d.relative_to(ctx.root)} already exists")
    (d / "runs").mkdir(parents=True)
    (d / "runs" / ".gitkeep").touch()
    title = (a.title or a.study.replace("-", " ").capitalize()).replace('"', "'")
    (d / "study.yaml").write_text(STUDY_TEMPLATE.format(
        id=a.study, title=title, ws=a.workstream, owner=owner,
        today=dt.date.today().isoformat()), encoding="utf-8")
    (d / "LOG.md").write_text(LOG_PREAMBLE.format(id=a.study), encoding="utf-8")
    print(f"created {d.relative_to(ctx.root)}/ (study.yaml, LOG.md, runs/). "
          f"Fill in the TODO fields in study.yaml before the first run.")


# --------------------------------------------------------------------------- run

def gpu_name():
    out = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def cmd_run(ctx: Ctx, a):
    if not a.command:
        die("give the command after `--`, e.g. record.py run s1 --tag smoke -- python train.py")
    if not TAG_RE.match(a.tag):
        die(f"tag '{a.tag}' may contain only letters, digits, '.', '_' and '-'")
    study = ctx.require_study(a.study)
    if study.get("status") == "planned":
        print("record.py: note: study status is still 'planned'; set it to 'running' in study.yaml.",
              file=sys.stderr)
    author = default_author(ctx, a.author)
    host = os.environ.get("NP_HOST") or socket.gethostname().split(".")[0]
    host = re.sub(r"[^A-Za-z0-9-]", "-", host)
    start = now_local()
    run_id = f"{start:%Y%m%d-%H%M}-{host}-{a.tag}"
    out_root = Path(os.path.expanduser(os.environ.get("NP_OUTPUTS", str(ctx.root / "outputs"))))
    run_dir = (out_root / ctx.repo / a.study / run_id).resolve()
    if run_dir.exists():
        die(f"{run_dir} already exists; wait a minute or use a different --tag")
    run_dir.mkdir(parents=True)

    commit, branch, dirty = git_info(ctx.root)
    command = " ".join(a.command)
    rec = {
        "id": run_id, "study": a.study, "author": author, "host": host,
        "started": start.isoformat(), "finished": None, "exit_code": None,
        "duration_min": None, "commit": commit, "branch": branch, "dirty": dirty,
        "command": command, "cwd": os.getcwd(), "seed": a.seed, "backfilled": False,
        "outputs_dir": str(Path(os.path.expanduser(a.outputs_dir)).resolve())
        if a.outputs_dir else str(run_dir),
        "gpu": gpu_name(), "python": sys.version.split()[0], "torch": pkg_version("torch"),
        "notes": a.note,
    }
    rec = {k: v for k, v in rec.items() if v is not None or k in RUN_REQUIRED}
    run_yaml = run_dir / "run.yaml"
    dump_yaml(rec, run_yaml)
    if dirty:
        print("record.py: warning: the working tree has uncommitted changes (dirty: true).",
              file=sys.stderr)
    print(f"record.py: run {run_id} -> {run_dir}", file=sys.stderr)

    env = dict(os.environ, NP_RUN_DIR=str(run_dir), NP_RUN_ID=run_id, NP_STUDY=a.study)
    if a.seed is not None:
        env["NP_SEED"] = str(a.seed)
    code = None
    with open(run_dir / "console.log", "w", encoding="utf-8", errors="replace") as log:
        try:
            proc = subprocess.Popen(a.command, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        except OSError as e:
            log.write(f"record.py: could not start command: {e}\n")
            code = 127
        else:
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log.write(line)
                code = proc.wait()
            except KeyboardInterrupt:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                code = -2
                log.write("\nrecord.py: interrupted by user\n")
    end = now_local()
    rec["finished"] = end.isoformat()
    rec["exit_code"] = code
    rec["duration_min"] = round((end - start).total_seconds() / 60, 2)
    dump_yaml(rec, run_yaml)
    print(f"record.py: run {run_id} finished with exit code {code} "
          f"after {rec['duration_min']} min", file=sys.stderr)
    return 0 if code == 0 else (code if isinstance(code, int) and code > 0 else 1)


# --------------------------------------------------------------------------- harvest

def rsync_small(src: str, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    args = ["rsync", "-a", "--prune-empty-dirs", f"--max-size={HARVEST_MAX}",
            "-e", "ssh -o BatchMode=yes", "--include=*/"]
    args += [f"--include={p}" for p in HARVEST_EXT] + ["--exclude=*", src.rstrip("/") + "/",
                                                       str(dest) + "/"]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode not in (0, 23, 24):   # 23/24: partial transfer (missing source etc.)
        raise RuntimeError(r.stderr.strip() or f"rsync exit {r.returncode}")
    return r.returncode


def cmd_harvest(ctx: Ctx, a):
    if shutil.which("rsync") is None:
        die("harvest needs rsync")
    source = a.source or os.environ.get("NP_OUTPUTS") or str(ctx.root / "outputs")
    remote = ":" in source and not Path(source).exists()
    host_prefix = source.split(":", 1)[0] + ":" if remote else ""
    base = source.rstrip("/") + "/" + ctx.repo
    if a.study:
        ctx.require_study(a.study)
        base += "/" + a.study
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        try:
            rsync_small(base, stage)
        except RuntimeError as e:
            die(f"rsync from {base} failed: {e}")
        study_dirs = [stage] if a.study else [p for p in stage.iterdir() if p.is_dir()]
        n = 0
        for sd in study_dirs:
            study = a.study or sd.name
            if not (ctx.study_dir(study) / "study.yaml").exists():
                print(f"record.py: skip runs of unknown study '{study}'", file=sys.stderr)
                continue
            for rd in sorted(p for p in sd.iterdir() if (p / "run.yaml").exists()):
                dest = ctx.study_dir(study) / "runs" / rd.name
                dest.mkdir(parents=True, exist_ok=True)
                for f in rd.rglob("*"):
                    if f.is_file() and f.name != "console.log":
                        t = dest / f.relative_to(rd)
                        t.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, t)
                con = rd / "console.log"
                if con.exists():
                    lines = con.read_text(encoding="utf-8", errors="replace").splitlines()
                    (dest / "console.tail.log").write_text(
                        "\n".join(lines[-CONSOLE_TAIL_LINES:]) + "\n", encoding="utf-8")
                rec = load_yaml(rd / "run.yaml") or {}
                od = rec.get("outputs_dir")
                if od and Path(od).name != rd.name:   # legacy script wrote elsewhere
                    try:
                        rsync_small(host_prefix + od, dest / "outputs")
                    except RuntimeError as e:
                        print(f"record.py: could not fetch outputs_dir {od}: {e}", file=sys.stderr)
                n += 1
                print(f"harvested {study}/{rd.name}")
        print(f"record.py: {n} run(s) harvested. Large files (>{HARVEST_MAX}) stay on the host.")


# --------------------------------------------------------------------------- log

LOG_ENTRY = """\
# {date} — <headline: the main finding, in one sentence>
Study: {study} · Author: {author} · Host: {host} · Commit: {commit} · Runs: {runs}

## Run
| What | Command or run id | Result |
|---|---|---|
| TODO | TODO | TODO |

## Findings
- TODO: what we saw, in full sentences, with numbers and their split (validation or test).

## Next
- [ ] TODO

"""


def parse_log(text: str):
    """Return (preamble_end_index, entries[(lineno, date, headline, body_lines)])."""
    lines = text.splitlines()
    entries, cur = [], None
    first = len(lines)
    for i, ln in enumerate(lines):
        m = LOG_HEAD_RE.match(ln)
        if m:
            if cur:
                entries.append(cur)
            cur = [i + 1, as_date(m.group(1)), m.group(3), []]
            first = min(first, i)
        elif cur:
            cur[3].append(ln)
    if cur:
        entries.append(cur)
    return first, entries, lines


def cmd_log(ctx: Ctx, a):
    ctx.require_study(a.study)
    f = ctx.study_dir(a.study) / "LOG.md"
    text = f.read_text(encoding="utf-8") if f.exists() else LOG_PREAMBLE.format(id=a.study)
    today = dt.date.today()
    runs_dir = ctx.study_dir(a.study) / "runs"
    todays = sorted(p.name for p in runs_dir.glob(f"{today:%Y%m%d}-*") if p.is_dir()) \
        if runs_dir.exists() else []
    commit, _, _ = git_info(ctx.root)
    entry = LOG_ENTRY.format(date=today.isoformat(), study=a.study,
                             author=default_author(ctx, a.author),
                             host=os.environ.get("NP_HOST") or socket.gethostname().split(".")[0],
                             commit=commit, runs=", ".join(todays) or "none")
    first, _, lines = parse_log(text)
    new = "\n".join(lines[:first]).rstrip() + "\n\n" + entry + "\n".join(lines[first:])
    f.write_text(new.rstrip() + "\n", encoding="utf-8")
    print(f"added an entry to {f.relative_to(ctx.root)}; replace every TODO and <...> placeholder.")


# --------------------------------------------------------------------------- check

class Report:
    def __init__(self, root: Path):
        self.root, self.errors, self.warnings = root, [], []

    def _fmt(self, path, field, msg):
        p = path.relative_to(self.root) if path.is_absolute() and self.root in path.parents else path
        return f"{p}: {field + ': ' if field else ''}{msg}"

    def err(self, path, field, msg):
        self.errors.append(self._fmt(path, field, msg))

    def warn(self, path, field, msg):
        self.warnings.append(self._fmt(path, field, msg))


def _need(rep, path, d, keys):
    for k in keys:
        if k not in d:
            rep.err(path, k, "missing (required)")
        elif d[k] in ("", []) :
            rep.err(path, k, "is empty")


def _placeholder(rep, path, d, keys, strict_err=True):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and PLACEHOLDER_RE.search(v):
            (rep.err if strict_err else rep.warn)(path, k, "still contains a TODO placeholder")


def check_people(ctx, rep, path, field, names):
    if ctx.people is None:
        return
    for n in names if isinstance(names, list) else [names]:
        if ctx.person(n) is None:
            rep.err(path, field, f"'{n}' is not in standards/people.yaml")


def check_key_results(rep, path, krs):
    if not isinstance(krs, list):
        rep.err(path, "key_results", "must be a list")
        return
    for i, kr in enumerate(krs):
        f = f"key_results[{i}]"
        if not isinstance(kr, dict):
            rep.err(path, f, "must be a mapping")
            continue
        _need(rep, path, kr, ["label", "value", "unit", "split", "n_seeds"])
        if "split" in kr and kr["split"] not in SPLITS:
            rep.err(path, f + ".split", f"must be one of {SPLITS} (say which data the number is from)")
        if "n_seeds" in kr and not (isinstance(kr["n_seeds"], int) and kr["n_seeds"] >= 1):
            rep.err(path, f + ".n_seeds", "must be a positive integer")
        if "value" in kr and not isinstance(kr["value"], (int, float)):
            rep.err(path, f + ".value", "must be a number")
        c = kr.get("compare")
        if c is not None and not (isinstance(c, dict) and "label" in c and "value" in c):
            rep.err(path, f + ".compare", "must have label and value")


def check_study(ctx, rep, sdir: Path):
    f = sdir / "study.yaml"
    try:
        s = load_yaml(f) or {}
    except yaml.YAMLError as e:
        rep.err(f, "", f"invalid YAML: {e}")
        return None
    if not isinstance(s, dict):
        rep.err(f, "", "must be a mapping")
        return None
    _need(rep, f, s, [k for k in STUDY_REQUIRED if k != "code"])
    if "code" not in s:
        rep.err(f, "code", "missing (required; use [] for a study without code)")
    if s.get("id") != sdir.name:
        rep.err(f, "id", f"must equal the folder name '{sdir.name}'")
    if s.get("id") and not SLUG_RE.match(str(s["id"])):
        rep.err(f, "id", "must be lowercase letters, digits and hyphens")
    if ctx.ws_ids() is not None and s.get("workstream") not in ctx.ws_ids():
        rep.err(f, "workstream", f"'{s.get('workstream')}' is not one of {ctx.ws_ids()}")
    st = s.get("status")
    if st not in STATUSES:
        rep.err(f, "status", f"'{st}' must be one of {STATUSES}")
    if s.get("criterion") not in CRITERIA:
        rep.err(f, "criterion", f"must be one of {CRITERIA}")
    if "owner" in s:
        if not isinstance(s["owner"], list):
            rep.err(f, "owner", "must be a list of names")
        else:
            check_people(ctx, rep, f, "owner", s["owner"])
    for k in ("started", "closed"):
        if k in s and as_date(s[k]) is None:
            rep.err(f, k, "must be a date YYYY-MM-DD")
    if "code" in s and not isinstance(s["code"], list):
        rep.err(f, "code", "must be a list of paths")
    for p in s.get("code") or []:
        if not (ctx.root / str(p)).exists():
            rep.warn(f, "code", f"path '{p}' does not exist")
    if st in CLOSED or st == "abandoned":
        _need(rep, f, s, ["closed", "verdict"])
        if st in CLOSED:
            _need(rep, f, s, ["key_results"])
            if not (sdir / "FINDINGS.md").exists():
                rep.err(sdir / "FINDINGS.md", "", "a closed study needs FINDINGS.md")
    if "key_results" in s:
        check_key_results(rep, f, s["key_results"])
    if st != "planned":
        _placeholder(rep, f, s, ["summary", "question", "hypothesis", "success_criterion", "verdict"])
    else:
        _placeholder(rep, f, s, ["summary", "question", "hypothesis", "success_criterion"],
                     strict_err=False)
    if as_date(s.get("started")) and as_date(s.get("closed")) and \
            as_date(s["closed"]) < as_date(s["started"]):
        rep.err(f, "closed", "is before started")
    return s


def check_run(ctx, rep, study_id, rdir: Path):
    f = rdir / "run.yaml"
    try:
        r = load_yaml(f) or {}
    except yaml.YAMLError as e:
        rep.err(f, "", f"invalid YAML: {e}")
        return
    _need_keys = [k for k in RUN_REQUIRED if k not in ("finished", "exit_code", "seed")]
    _need(rep, f, r, _need_keys)
    for k in ("finished", "exit_code", "seed"):
        if k not in r:
            rep.err(f, k, "missing (required; may be null)")
    if r.get("id") != rdir.name:
        rep.err(f, "id", f"must equal the folder name '{rdir.name}'")
    elif not RUN_ID_RE.match(str(r["id"])):
        rep.err(f, "id", "must look like YYYYMMDD-HHMM-<host>-<tag>")
    if r.get("study") != study_id:
        rep.err(f, "study", f"must be '{study_id}'")
    if not isinstance(r.get("backfilled"), bool):
        rep.err(f, "backfilled", "must be true or false")
    if r.get("backfilled") and not r.get("source"):
        rep.err(f, "source", "a backfilled run must say where its information came from")
    if "author" in r:
        check_people(ctx, rep, f, "author", r["author"])
    if r.get("finished") is None or r.get("exit_code") is None:
        rep.warn(f, "finished", "run has no finish time or exit code (still running or crashed)")
    elif r.get("exit_code") != 0:
        rep.warn(f, "exit_code", f"run failed with exit code {r['exit_code']}")
    if r.get("dirty"):
        rep.warn(f, "dirty", "run used uncommitted code")


def check_log(ctx, rep, f: Path, study_id):
    text = f.read_text(encoding="utf-8")
    _, entries, _ = parse_log(text)
    prev = None
    for lineno, date, headline, body in entries:
        loc = f"line {lineno}"
        if date is None:
            rep.err(f, loc, "bad date in heading")
            continue
        if prev and date > prev:
            rep.err(f, loc, f"entry {date} is newer than the one above it; keep newest first")
        prev = date
        if date < ADOPTION_DATE:
            continue          # legacy entries keep their original format
        if PLACEHOLDER_RE.search(headline):
            rep.err(f, loc, "headline is still a placeholder")
        head = next((ln for ln in body if ln.strip()), "")
        if not head.startswith("Study:"):
            rep.err(f, loc, "first line after the heading must be "
                            "'Study: · Author: · Host: · Commit: · Runs:'")
        else:
            m = re.search(r"Study:\s*([^·]+)", head)
            if m and m.group(1).strip() != study_id:
                rep.err(f, loc, f"Study: says '{m.group(1).strip()}', expected '{study_id}'")
            m = re.search(r"Author:\s*([^·]+)", head)
            if m:
                check_people(ctx, rep, f, loc + " Author", m.group(1).strip())
        sections = {ln.strip() for ln in body if ln.startswith("## ")}
        for sec in ("## Run", "## Findings", "## Next"):
            if sec not in sections:
                rep.err(f, loc, f"missing section '{sec}'")
        if any("TODO" in ln for ln in body):
            rep.err(f, loc, "entry still contains TODO placeholders")


def check_md_images(rep, md: Path, body: str):
    for target in IMG_RE.findall(body):
        if re.match(r"^[a-z]+://", target) or target.startswith("/"):
            continue
        if not (md.parent / target.split("#")[0]).exists():
            rep.err(md, "image", f"'{target}' does not exist")


def build_manifest(ctx: Ctx):
    studies, posts, ws = [], [], set()
    for p in sorted((ctx.showcase / "studies").glob("*.md")):
        fm, _ = split_frontmatter(p.read_text(encoding="utf-8"))
        if not fm:
            continue
        ws.add(fm.get("workstream"))
        studies.append({"id": fm.get("id"), "title": fm.get("title"),
                        "workstream": fm.get("workstream"), "status": fm.get("status"),
                        "summary": fm.get("summary"), "page": f"studies/{p.name}"})
    for p in sorted((ctx.showcase / "posts").glob("*.md")):
        fm, _ = split_frontmatter(p.read_text(encoding="utf-8"))
        if not fm:
            continue
        posts.append({"file": f"posts/{p.name}", "title": fm.get("title"),
                      "date": str(fm.get("date")), "draft": bool(fm.get("draft", False))})
    return {"repo": ctx.repo, "generated_by": f"record.py {VERSION}",
            "workstreams": sorted(w for w in ws if w), "studies": studies, "posts": posts}


def check_showcase(ctx, rep):
    sc = ctx.showcase
    if not sc.exists():
        return
    for p in sorted((sc / "studies").glob("*.md")):
        fm, body = split_frontmatter(p.read_text(encoding="utf-8"))
        if fm is None:
            rep.err(p, "", "missing YAML frontmatter")
            continue
        _need(rep, p, fm, PAGE_REQUIRED)
        if fm.get("status") not in STATUSES:
            rep.err(p, "status", f"must be one of {STATUSES}")
        if fm.get("repo") != ctx.repo:
            rep.err(p, "repo", f"must be '{ctx.repo}'")
        if "key_results" in fm:
            check_key_results(rep, p, fm["key_results"])
        check_md_images(rep, p, body)
    for p in sorted((sc / "posts").glob("*.md")):
        fm, body = split_frontmatter(p.read_text(encoding="utf-8"))
        if fm is None:
            rep.err(p, "", "missing YAML frontmatter")
            continue
        _need(rep, p, fm, POST_REQUIRED)
        if "authors" in fm:
            if not isinstance(fm["authors"], list):
                rep.err(p, "authors", "must be a list")
            else:
                check_people(ctx, rep, p, "authors", fm["authors"])
        if ctx.ws_ids() is not None and fm.get("workstream") not in ctx.ws_ids():
            rep.err(p, "workstream", f"'{fm.get('workstream')}' is not one of {ctx.ws_ids()}")
        if as_date(fm.get("date")) is None:
            rep.err(p, "date", "must be a date YYYY-MM-DD")
        if not re.match(r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md$", p.name):
            rep.err(p, "", "file name must be YYYY-MM-DD-<slug>.md")
        if not fm.get("draft") and PLACEHOLDER_RE.search(body + str(fm.get("summary", ""))):
            rep.err(p, "", "published post (draft: false) still contains TODO placeholders")
        check_md_images(rep, p, body)
    mf = sc / "manifest.yaml"
    if not mf.exists():
        rep.err(mf, "", "missing; run `record.py promote <study>` to generate it")
    elif (load_yaml(mf) or {}) != build_manifest(ctx):
        rep.err(mf, "", "out of date with showcase/; run `record.py promote <study>` "
                        "(or `record.py check --fix-manifest`)")


def cmd_check(ctx: Ctx, a):
    rep = Report(ctx.root)
    if ctx.people is None:
        rep.warn(ctx.root / "tools" / "standards", "", "people.yaml/workstreams.yaml not found; "
                 "names and workstreams are not validated (run `record.py self-update`)")
    if a.fix_manifest and ctx.showcase.exists():
        dump_yaml(build_manifest(ctx), ctx.showcase / "manifest.yaml")
    n_s = n_r = 0
    if ctx.studies.exists():
        for sdir in sorted(p for p in ctx.studies.iterdir() if p.is_dir()):
            if not (sdir / "study.yaml").exists():
                rep.err(sdir, "", "study folder without study.yaml")
                continue
            n_s += 1
            check_study(ctx, rep, sdir)
            if (sdir / "LOG.md").exists():
                check_log(ctx, rep, sdir / "LOG.md", sdir.name)
            else:
                rep.err(sdir / "LOG.md", "", "missing")
            runs = sdir / "runs"
            for rdir in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []:
                if not (rdir / "run.yaml").exists():
                    rep.err(rdir, "", "run folder without run.yaml")
                    continue
                n_r += 1
                check_run(ctx, rep, sdir.name, rdir)
    check_showcase(ctx, rep)
    for w in rep.warnings:
        print(f"warning: {w}")
    for e in rep.errors:
        print(f"error:   {e}")
    fail = bool(rep.errors) or (a.strict and bool(rep.warnings))
    print(f"record.py check: {n_s} studies, {n_r} runs, {len(rep.errors)} errors, "
          f"{len(rep.warnings)} warnings{' (strict)' if a.strict else ''} -> "
          f"{'FAIL' if fail else 'OK'}")
    return 1 if fail else 0


# --------------------------------------------------------------------------- promote

POST_STUB = """\
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
"""


def code_url(ctx, path):
    return f"https://github.com/{ORG}/{ctx.repo}/tree/main/{path}"


def cmd_promote(ctx: Ctx, a):
    s = ctx.require_study(a.study)
    rep = Report(ctx.root)
    check_study(ctx, rep, ctx.study_dir(a.study))
    if rep.errors:
        for e in rep.errors:
            print(f"error:   {e}")
        die("fix study.yaml before promoting", 1)
    if s["status"] == "planned":
        die("a planned study has nothing to show yet")
    sdir = ctx.study_dir(a.study)
    fig_dir = ctx.showcase / "figures" / a.study
    findings = sdir / "FINDINGS.md"
    body = findings.read_text(encoding="utf-8") if findings.exists() else \
        "*This study is still running. Results will appear here when it closes.*\n"

    def relink(m):
        target = m.group(1)
        src = (sdir / target)
        if re.match(r"^[a-z]+://", target) or not src.is_file():
            return m.group(0)
        name = re.sub(r"[^A-Za-z0-9_.-]", "-", target.replace("/", "--")).lstrip("-.")
        fig_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, fig_dir / name)
        return m.group(0).replace(target, f"../figures/{a.study}/{name}")

    body = IMG_RE.sub(relink, body)
    fm = {"id": s["id"], "repo": ctx.repo, "title": s["title"], "workstream": s["workstream"],
          "status": s["status"], "owner": s["owner"], "started": str(as_date(s["started"])),
          "closed": str(as_date(s["closed"])) if s.get("closed") else None,
          "summary": s["summary"], "question": s["question"], "hypothesis": s["hypothesis"],
          "success_criterion": s["success_criterion"], "criterion": s["criterion"],
          "verdict": s.get("verdict"), "key_results": s.get("key_results"),
          "code_url": code_url(ctx, f"studies/{a.study}")}
    fm = {k: v for k, v in fm.items() if v is not None}
    page = ctx.showcase / "studies" / f"{a.study}.md"
    note = (f"<!-- Generated by record.py promote from studies/{a.study}/study.yaml and "
            f"FINDINGS.md. Edit those files, not this one. -->\n\n")
    write_frontmatter(page, fm, note + body)
    print(f"wrote {page.relative_to(ctx.root)}")

    posts = ctx.showcase / "posts"
    has_post = False
    for p in posts.glob("*.md") if posts.exists() else []:
        pfm, _ = split_frontmatter(p.read_text(encoding="utf-8"))
        if pfm and pfm.get("study") == a.study:
            has_post = True
    if not has_post and s["status"] != "running":
        today = dt.date.today().isoformat()
        post = posts / f"{today}-{a.study}.md"
        write_frontmatter(post, {"title": s["title"], "date": today, "authors": s["owner"],
                                 "workstream": s["workstream"], "study": a.study,
                                 "summary": s["summary"], "draft": True}, POST_STUB)
        print(f"wrote draft post {post.relative_to(ctx.root)} (draft: true until edited)")
    dump_yaml(build_manifest(ctx), ctx.showcase / "manifest.yaml")
    print(f"updated {(ctx.showcase / 'manifest.yaml').relative_to(ctx.root)}")


# --------------------------------------------------------------------------- self-update

def cmd_self_update(ctx: Ctx, a):
    targets = {"tools/record.py": Path(__file__).resolve(),
               "standards/workstreams.yaml": ctx.root / "tools" / "standards" / "workstreams.yaml",
               "standards/people.yaml": ctx.root / "tools" / "standards" / "people.yaml"}
    for remote, local in targets.items():
        url = f"{RAW_BASE}/{remote}"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                new = r.read().decode("utf-8")
        except Exception as e:
            print(f"could not fetch {url}: {e}", file=sys.stderr)
            continue
        old = local.read_text(encoding="utf-8") if local.exists() else ""
        if old == new:
            print(f"{remote}: up to date")
            continue
        sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                   f"local/{remote}", f"upstream/{remote}"))
        if a.yes or input(f"\nreplace {local}? [y/N] ").strip().lower() == "y":
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(new, encoding="utf-8")
            print(f"{remote}: updated")


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(prog="record.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", help="repo root (default: git toplevel of the current directory)")
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="scaffold studies/<study>/")
    p.add_argument("study")
    p.add_argument("--workstream", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--title")

    p = sub.add_parser("run", help="run a command and record it")
    p.add_argument("study")
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--outputs-dir", help="where a legacy script writes its outputs")
    p.add_argument("--author")
    p.add_argument("--note")

    p = sub.add_parser("harvest", help="copy run records and small outputs into studies/")
    p.add_argument("study", nargs="?")
    p.add_argument("--from", dest="source",
                   help="outputs root, local or user@host:/abs/path (default: $NP_OUTPUTS)")

    p = sub.add_parser("log", help="prepend a LOG.md entry template")
    p.add_argument("study")
    p.add_argument("--author")

    p = sub.add_parser("check", help="validate all records")
    p.add_argument("--strict", action="store_true", help="treat warnings as errors")
    p.add_argument("--fix-manifest", action="store_true", help="regenerate showcase/manifest.yaml")

    p = sub.add_parser("promote", help="publish a study to showcase/")
    p.add_argument("study")

    p = sub.add_parser("self-update", help="update this file and the standards from upstream")
    p.add_argument("--yes", action="store_true")

    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if "--" in argv:                       # everything after `--` is the command to run
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    a = ap.parse_args(argv)
    a.command = command
    ctx = Ctx(find_root(a.root))
    fn = {"new": cmd_new, "run": cmd_run, "harvest": cmd_harvest, "log": cmd_log,
          "check": cmd_check, "promote": cmd_promote, "self-update": cmd_self_update}[a.cmd]
    return fn(ctx, a) or 0


if __name__ == "__main__":
    sys.exit(main())
