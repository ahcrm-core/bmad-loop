"""Opt-in runtime metrics for this test suite: ``--test-metrics-dir=PATH``.

The Windows CI legs take tens of minutes and have failed in ways a pytest summary
line cannot explain: a killed job leaves no summary at all, ``--durations`` names
the slowest phases but not where the rest of the time went, and cumulative time
spread across xdist workers is easy to mistake for elapsed time. This module
records enough to answer those questions from files, including for a run that
never finished.

**Inert by default.** ``tests/conftest.py`` only registers the two options below.
Without ``--test-metrics-dir`` no plugin object is registered, no hook of this
module runs, no audit hook is installed and nothing is written, so ordinary
pytest behavior is unchanged. Spell the option with ``=``: ``pytest -q
--test-metrics-dir PATH`` with no test path makes pytest's early argument scan
take ``PATH`` for a test path, skip ``testpaths`` and never load
``tests/conftest.py``, so the option then reads as unrecognized.

**One run per directory.** The controlling process refuses a directory that
already holds a ``manifest.json``. Files written there:

``manifest.json``
    Once, by the controlling process (the xdist controller, or the only process of
    a run without ``-n``): schema version, run id, commit, interpreter, platform,
    worker count, selection arguments relative to the rootdir, and the temp
    directory's drive anchor.
``events-<worker>-<pid>.jsonl``
    One per pytest process (``main``, ``controller``, ``gw0``, ...). Append-only,
    one JSON object per line, flushed after every line, so a killed run keeps
    every record up to its last event and names the test each worker had started
    but not finished. The pid in the name keeps a restarted xdist worker from
    reusing its predecessor's file; files are opened exclusively.
``inventory-<worker>-<pid>.json``
    From each collecting process: counts and digests of the collected and the
    final selected nodeids, kept apart. xdist's controller does not collect and
    every worker collects the whole suite, so only ``main`` or ``gw0`` also
    writes the full collected, selected and deselected lists; the summary checks
    every other worker's digests against them.
``runner.json``
    Only from ``scripts/run_test_benchmark.py``: the pytest process lifetime,
    exit code, and whether the wrapper's deadline killed it.
``summary.json`` / ``summary.txt``
    Derived; regenerate with ``python tests/perf_report.py summarize PATH``.

Nodeids are canonical: xdist's ``loadgroup`` scheduler rewrites a grouped test's
id to ``<id>@<group>`` on workers only; the suffix is stripped so inventories from
runs with and without ``-n`` compare equal.

Event kinds (every record carries ``v``, ``kind`` and ``mono``, a
``time.monotonic()`` reading comparable across the processes of one machine):
``session_start``, ``collection``, ``start`` (a test began), ``phase`` (one
setup/call/teardown report: outcome, ``duration`` and pytest's wall-clock
``start``/``stop``), ``spawns`` (subprocess launches during one test, attribution
mode only), ``totals`` (cumulative fixture setup and subprocess counts, re-emitted
every ``TOTALS_INTERVAL_S`` and at session end, so a killed run keeps a recent
snapshot), ``node_down`` (controller: an xdist worker went away),
``sessionfinish_start``/``session_finish`` (bracketing pytest's own sessionfinish
work: JUnit, terminal summary, xdist teardown), ``unconfigure`` and ``exit`` (an
``atexit`` handler registered at session start, so it runs after pytest's later
registered temp-directory cleanup).

**Two clocks, never summed into each other.** A test phase's ``duration`` is
worker time. Summed across workers it is *worker-seconds*, which exceeds elapsed
time whenever workers overlap. Elapsed time comes only from monotonic readings on
one process (the controller's session start to finish) or from the wrapper's
process lifetime. The summary labels the two separately.

**Subprocess attribution** (``--test-metrics-subprocesses``, off by default and
off in CI) installs a ``sys.audit`` hook that counts ``subprocess.Popen`` launches
per test, per command and per call site. It pays a Python call for every audited
event in the process, so it is a targeted diagnostic, not a default. Launches by
children (git's own grandchildren, xdist workers' descendants) are not counted.

**Privacy.** Nothing here records environment variables, test output, captured
logs, failure text, or argv beyond a program's base name and, for git, its
subcommand verb. Paths are recorded only as rootdir-relative selection arguments
and the temp directory's drive anchor; call sites are repository-relative.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from collections import Counter
from collections.abc import Iterable, Sequence
from importlib import metadata
from pathlib import Path
from types import FrameType
from typing import Any

import pytest

SCHEMA_VERSION = 1
PLUGIN_NAME = "bmad-loop-test-metrics"

MANIFEST = "manifest.json"
RUNNER = "runner.json"
SUMMARY_JSON = "summary.json"
SUMMARY_TEXT = "summary.txt"
EVENTS_PREFIX = "events-"
INVENTORY_PREFIX = "inventory-"

# How often a worker re-emits its cumulative fixture/subprocess totals. A killed
# run loses at most this much of that (and nothing of the per-test records).
TOTALS_INTERVAL_S = 30.0
TOP_N = 25

_REPO = Path(__file__).resolve().parent.parent
_PROJECT_DIRS = (str(_REPO / "src" / "bmad_loop") + os.sep, str(_REPO / "tests") + os.sep)
_THIS_FILE = str(Path(__file__).resolve())

# Keys this plugin adds to each xdist worker's `workerinput`.
_RUN_ID_KEY = "bmad_loop_metrics_run_id"
_ROOT_KEY = "bmad_loop_metrics_root"

EXIT_STATUS_NAMES = {
    0: "passed",
    1: "tests_failed",
    2: "interrupted",
    3: "internal_error",
    4: "usage_error",
    5: "no_tests_collected",
}


class MetricsDegradedWarning(pytest.PytestWarning):
    """The metrics plugin stopped recording (e.g. its output file became unwritable)."""


# --------------------------------------------------------------------- options


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("bmad-loop test metrics")
    group.addoption(
        "--test-metrics-dir",
        dest="test_metrics_dir",
        default=None,
        metavar="PATH",
        help="record per-worker runtime metrics under PATH (a fresh directory; "
        "spell it --test-metrics-dir=PATH). See tests/perf_report.py.",
    )
    group.addoption(
        "--test-metrics-subprocesses",
        dest="test_metrics_subprocesses",
        action="store_true",
        default=False,
        help="with --test-metrics-dir: also count subprocess launches per test, "
        "command and call site (diagnostic; adds overhead).",
    )


def configure(config: pytest.Config) -> None:
    raw = config.getoption("test_metrics_dir")
    subprocesses = bool(config.getoption("test_metrics_subprocesses"))
    if raw is None:
        if subprocesses:
            raise pytest.UsageError("--test-metrics-subprocesses requires --test-metrics-dir")
        return
    workerinput: dict[str, Any] | None = getattr(config, "workerinput", None)
    if workerinput is None:
        root = (config.invocation_params.dir / raw).resolve()
        _claim_directory(root)
        run_id = uuid.uuid4().hex
        worker = None
    else:
        root = Path(workerinput.get(_ROOT_KEY) or (config.invocation_params.dir / raw)).resolve()
        run_id = str(workerinput.get(_RUN_ID_KEY, "unknown"))
        worker = str(workerinput["workerid"])
    plugin = MetricsPlugin(config, root, run_id=run_id, worker=worker, subprocesses=subprocesses)
    config.pluginmanager.register(plugin, PLUGIN_NAME)


def _claim_directory(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root / MANIFEST).exists():
        raise pytest.UsageError(
            f"--test-metrics-dir {root} already holds a metrics run ({MANIFEST}); "
            "pass a fresh directory so two runs' records cannot mix"
        )


# --------------------------------------------------------------------- writing


class _EventWriter:
    """Append-only JSONL, flushed per line. A write failure stops recording and
    warns once; it never fails the test that happened to be running."""

    def __init__(self, handle: Any, path: Path) -> None:
        self._handle = handle
        self.path = path
        self.failure: str | None = None

    @classmethod
    def create(cls, root: Path, name: str) -> _EventWriter:
        pid = os.getpid()
        for attempt in range(100):
            suffix = f"-{attempt}" if attempt else ""
            path = root / f"{EVENTS_PREFIX}{name}-{pid}{suffix}.jsonl"
            try:
                handle = path.open("x", encoding="utf-8", newline="\n")
            except FileExistsError:
                continue
            return cls(handle, path)
        raise pytest.UsageError(f"cannot create a fresh metrics events file in {root}")

    def emit(self, record: dict[str, Any]) -> None:
        if self.failure is not None:
            return
        try:
            self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._handle.flush()
        except (OSError, ValueError) as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                MetricsDegradedWarning(
                    f"test metrics: stopped writing {self.path.name}: {self.failure}"
                ),
                stacklevel=2,
            )

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


def _write_json_atomic(path: Path, payload: Any, *, compact: bool = False) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = (
        json.dumps(payload, separators=(",", ":"))
        if compact
        else json.dumps(payload, indent=1, sort_keys=True)
    )
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------ command naming

_SAFE_NAME = re.compile(r"[A-Za-z0-9._+-]{1,64}")
_PYTHON_NAME = re.compile(r"python(\d+(\.\d+)?)?w?")
_GIT_VERB = re.compile(r"[a-z][a-z0-9-]{0,39}")
_GIT_OPTIONS_WITH_VALUE = frozenset(
    {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--super-prefix"}
)


def program_name(token: str) -> str:
    """A launched program's base name, stripped of Windows executable suffixes.
    Anything that does not look like a plain name is reported as ``<other>`` so a
    path or an argument can never leak through this field."""
    name = token.strip().strip("\"'").replace("\\", "/").rsplit("/", 1)[-1]
    lowered = name.lower()
    for suffix in (".exe", ".cmd", ".bat", ".com"):
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            lowered = lowered[: -len(suffix)]
            break
    if _PYTHON_NAME.fullmatch(lowered):
        return "python"
    return name if _SAFE_NAME.fullmatch(name) else "<other>"


def git_verb(args: Sequence[str]) -> str:
    """The subcommand of a git argv (after the program), skipping global options."""
    skip = False
    for token in args:
        if skip:
            skip = False
            continue
        if token in _GIT_OPTIONS_WITH_VALUE:
            skip = True
            continue
        if token.startswith("-"):
            continue
        return token if _GIT_VERB.fullmatch(token) else "<other>"
    return "<no-verb>"


def describe_command(executable: object, args: object) -> str:
    """``program`` or ``git <verb>`` for a ``subprocess.Popen`` audit event."""
    if isinstance(args, (str, bytes, os.PathLike)):
        text = os.fsdecode(args).strip()
        if text[:1] in ('"', "'"):
            end = text.find(text[0], 1)
            argv = [text[1:end] if end > 0 else text[1:]]
        else:
            argv = text.split()[:1]
    elif isinstance(args, Iterable):
        argv = [os.fsdecode(a) if isinstance(a, (str, bytes, os.PathLike)) else "" for a in args]
    else:
        argv = []
    first = (
        argv[0]
        if argv
        else (os.fsdecode(executable) if isinstance(executable, (str, bytes)) else "")
    )
    program = program_name(first)
    if program == "git":
        return f"git {git_verb(argv[1:])}"
    return program


def call_site(frame: FrameType | None, depth: int = 3) -> str:
    """The nearest ``depth`` repository frames (src/bmad_loop or tests), innermost
    first, as ``relpath:qualname`` joined by `` < ``."""
    parts: list[str] = []
    while frame is not None and len(parts) < depth:
        filename = frame.f_code.co_filename
        if filename != _THIS_FILE and filename.startswith(_PROJECT_DIRS):
            rel = Path(filename).relative_to(_REPO).as_posix()
            parts.append(f"{rel}:{frame.f_code.co_qualname}")
        frame = frame.f_back
    return " < ".join(parts) if parts else "<external>"


# ----------------------------------------------------------------- the plugin


def _canonical_nodeid(item: pytest.Item) -> str:
    """Strip xdist's ``@<group>`` loadgroup suffix (added on workers only)."""
    nodeid = item.nodeid
    names = sorted(
        {
            str(m.args[0] if m.args else m.kwargs.get("name", "default"))
            for m in item.iter_markers("xdist_group")
        }
    )
    suffix = "@" + "_".join(names)
    if names and nodeid.endswith(suffix):
        return nodeid[: -len(suffix)]
    return nodeid


def _relative_selection(arg: str, invocation_dir: Path, rootpath: Path) -> str:
    path_part, sep, rest = str(arg).partition("::")
    try:
        rel = (invocation_dir / path_part).resolve().relative_to(rootpath).as_posix()
    except (ValueError, OSError):
        return "<outside-rootdir>"
    return f"{rel}{sep}{rest}"


def _git_head(cwd: Path) -> tuple[str | None, str | None]:
    try:
        done = subprocess.run(  # fixed argv, no shell; test tooling, not src/
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, type(exc).__name__
    if done.returncode != 0:
        return None, f"git exited {done.returncode}"
    return done.stdout.strip() or None, None


def nodeid_digest(nodeids: Iterable[str]) -> str:
    """Order-independent digest of a nodeid set, for cross-worker comparison."""
    return hashlib.sha256("\n".join(sorted(nodeids)).encode("utf-8")).hexdigest()


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class MetricsPlugin:
    """Per-process recorder. Registered only when ``--test-metrics-dir`` is given."""

    def __init__(
        self,
        config: pytest.Config,
        root: Path,
        *,
        run_id: str,
        worker: str | None,
        subprocesses: bool,
    ) -> None:
        self._config = config
        self.root = root
        self.run_id = run_id
        self._worker = worker
        self._subprocesses = subprocesses
        self._configured_mono = time.monotonic()
        self._name = worker or "main"
        self._role = "worker" if worker else "main"
        self._writer: _EventWriter | None = None
        self._fixtures: dict[tuple[str, str], list[float]] = {}
        self._canonical: dict[str, str] = {}
        self._collected: list[str] = []
        self._deselected: list[str] = []
        self._collect_errors = 0
        self._active: str | None = None
        self._last_totals = time.monotonic()
        self._reports_seen = 0
        self._last_report_mono: float | None = None
        self._spawns_test: Counter[str] = Counter()
        self._spawns_by_command: Counter[str] = Counter()
        self._spawns_by_site: Counter[str] = Counter()
        self._spawn_hook_errors = 0

    # -- plumbing

    def _emit(self, kind: str, **fields: Any) -> None:
        if self._writer is None:
            return
        record: dict[str, Any] = {"v": SCHEMA_VERSION, "kind": kind, "mono": time.monotonic()}
        record.update(fields)
        self._writer.emit(record)

    def _canon(self, nodeid: str) -> str:
        return self._canonical.get(nodeid, nodeid)

    def _emit_totals(self) -> None:
        self._last_totals = time.monotonic()
        fields: dict[str, Any] = {
            "fixtures": [
                [name, scope, int(slot[0]), round(slot[1], 6)]
                for (name, scope), slot in sorted(self._fixtures.items())
            ]
        }
        if self._subprocesses:
            fields["spawns"] = {
                "by_command": dict(self._spawns_by_command),
                "by_site": dict(self._spawns_by_site),
                "hook_errors": self._spawn_hook_errors,
            }
        self._emit("totals", **fields)

    def _audit(self, event: str, args: tuple[Any, ...]) -> None:
        if event != "subprocess.Popen":
            return
        # An exception escaping an audit hook aborts the audited operation, i.e.
        # it would fail the test's Popen. Count it instead; the count is recorded.
        try:
            command = describe_command(args[0], args[1])
            site = call_site(sys._getframe(1))
        except Exception:
            self._spawn_hook_errors += 1
            return
        self._spawns_by_command[command] += 1
        self._spawns_by_site[site] += 1
        self._spawns_test[command] += 1

    def _manifest(self) -> dict[str, Any]:
        config = self._config
        rootpath = config.rootpath
        invocation_dir = config.invocation_params.dir
        commit, commit_error = _git_head(rootpath)
        return {
            "v": SCHEMA_VERSION,
            "run_id": self.run_id,
            "role": self._role,
            "events_file": self._writer.path.name if self._writer else None,
            "created_wall": time.time(),
            "commit": commit,
            "commit_error": commit_error,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "os": platform.platform(terse=True),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "pytest": pytest.__version__,
            "pytest_xdist": _dist_version("pytest-xdist"),
            "numprocesses": config.getoption("numprocesses", default=None),
            "dist": config.getoption("dist", default=None),
            "selection": {
                "args": [_relative_selection(a, invocation_dir, rootpath) for a in config.args],
                "ignore": [
                    _relative_selection(a, invocation_dir, rootpath)
                    for a in (config.getoption("ignore") or [])
                ],
                "keyword_expression": bool(config.getoption("keyword", default="")),
                "mark_expression": bool(config.getoption("markexpr", default="")),
            },
            "temp_anchor": Path(tempfile.gettempdir()).anchor or "/",
            "utf8_mode": bool(sys.flags.utf8_mode),
            "subprocess_attribution": self._subprocesses,
            "junitxml": bool(getattr(config.option, "xmlpath", None)),
        }

    def _at_exit(self) -> None:
        self._emit("exit")
        if self._writer is not None:
            self._writer.close()

    # -- session

    @pytest.hookimpl(trylast=True)
    def pytest_sessionstart(self, session: pytest.Session) -> None:
        if self._worker is None and self._config.pluginmanager.getplugin("dsession") is not None:
            self._name = self._role = "controller"
        self._writer = _EventWriter.create(self.root, self._name)
        self._emit(
            "session_start",
            run_id=self.run_id,
            worker=self._name,
            role=self._role,
            pid=os.getpid(),
            configured_mono=self._configured_mono,
            wall=time.time(),
            python=platform.python_version(),
            platform=sys.platform,
            xdist_worker_count=(
                os.environ.get("PYTEST_XDIST_WORKER_COUNT") if self._worker else None
            ),
        )
        if self._role != "worker":
            _write_json_atomic(self.root / MANIFEST, self._manifest())
        atexit.register(self._at_exit)
        if self._subprocesses:
            sys.addaudithook(self._audit)

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        node.workerinput[_RUN_ID_KEY] = self.run_id
        node.workerinput[_ROOT_KEY] = str(self.root)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        workerinput = getattr(node, "workerinput", None) or {}
        self._emit(
            "node_down", worker=str(workerinput.get("workerid", "?")), crashed=error is not None
        )

    # -- collection

    @pytest.hookimpl(wrapper=True)
    def pytest_collection(self, session: pytest.Session) -> Any:
        started = time.monotonic()
        try:
            return (yield)
        finally:
            if self._role != "controller":
                self._finish_collection(session, started)

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(
        self, session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
    ) -> Any:
        self._collected = [_canonical_nodeid(item) for item in items]
        try:
            return (yield)
        finally:
            for item in items:
                canonical = _canonical_nodeid(item)
                if canonical != item.nodeid:
                    self._canonical[item.nodeid] = canonical

    def pytest_deselected(self, items: Sequence[pytest.Item]) -> None:
        self._deselected.extend(_canonical_nodeid(item) for item in items)

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self._collect_errors += 1

    def _finish_collection(self, session: pytest.Session, started: float) -> None:
        selected = [_canonical_nodeid(item) for item in session.items]
        name = f"{INVENTORY_PREFIX}{self._name}-{os.getpid()}.json"
        inventory: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "run_id": self.run_id,
            "worker": self._name,
            "counts": {
                "collected": len(self._collected),
                "selected": len(selected),
                "deselected": len(self._deselected),
            },
            "digests": {
                "collected": nodeid_digest(self._collected),
                "selected": nodeid_digest(selected),
            },
            "collect_errors": self._collect_errors,
        }
        # Every xdist worker collects the whole suite (xdist itself refuses to run
        # on differing collections), so one full copy is enough; the others carry
        # digests the summary checks against it.
        if self._name in ("main", "gw0"):
            inventory.update(
                collected=self._collected, selected=selected, deselected=self._deselected
            )
        try:
            _write_json_atomic(self.root / name, inventory, compact=True)
        except OSError as exc:
            name = None
            warnings.warn(
                MetricsDegradedWarning(f"test metrics: inventory not written: {exc}"), stacklevel=2
            )
        self._emit(
            "collection",
            started_mono=started,
            seconds=round(time.monotonic() - started, 6),
            collected=len(self._collected),
            selected=len(selected),
            deselected=len(self._deselected),
            collect_errors=self._collect_errors,
            inventory=name,
        )

    # -- tests

    def pytest_runtest_logstart(self, nodeid: str, location: object) -> None:
        if self._role == "controller":
            return
        self._active = self._canon(nodeid)
        self._spawns_test.clear()
        self._emit("start", nodeid=self._active)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if self._role == "controller":
            self._reports_seen += 1
            self._last_report_mono = time.monotonic()
            return
        fields: dict[str, Any] = {
            "nodeid": self._canon(report.nodeid),
            "when": report.when,
            "outcome": report.outcome,
            "duration": round(report.duration, 6),
            "start": report.start,
            "stop": report.stop,
        }
        if hasattr(report, "wasxfail"):
            fields["xfail"] = True
        self._emit("phase", **fields)

    def pytest_runtest_logfinish(self, nodeid: str, location: object) -> None:
        if self._role == "controller":
            return
        if self._subprocesses and self._spawns_test:
            self._emit(
                "spawns",
                nodeid=self._canon(nodeid),
                count=sum(self._spawns_test.values()),
                by_command=dict(self._spawns_test),
            )
        self._spawns_test.clear()
        self._active = None
        if time.monotonic() - self._last_totals >= TOTALS_INTERVAL_S:
            self._emit_totals()

    @pytest.hookimpl(wrapper=True)
    def pytest_fixture_setup(self, fixturedef: Any, request: pytest.FixtureRequest) -> Any:
        # Exclusive of the fixtures it requests: pytest resolves those before
        # calling this hook (FixtureDef.execute).
        started = time.perf_counter()
        try:
            return (yield)
        finally:
            slot = self._fixtures.setdefault(
                (str(fixturedef.argname), str(fixturedef.scope)), [0, 0.0]
            )
            slot[0] += 1
            slot[1] += time.perf_counter() - started

    # -- end of session

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> Any:
        self._emit("sessionfinish_start", exitstatus=int(exitstatus))
        try:
            return (yield)
        finally:
            self._emit_totals()
            self._emit(
                "session_finish",
                exitstatus=int(session.exitstatus),
                testsfailed=session.testsfailed,
                testscollected=session.testscollected,
                reports_seen=self._reports_seen if self._role == "controller" else None,
                last_report_mono=self._last_report_mono,
                writer_failure=self._writer.failure if self._writer else None,
            )

    @pytest.hookimpl(trylast=True)
    def pytest_unconfigure(self, config: pytest.Config) -> None:
        self._emit("unconfigure")
        if self._role == "worker":
            return
        try:
            write_summary(self.root)
        except (OSError, ValueError) as exc:
            # The run's own records are intact; only the derived view is missing.
            sys.stderr.write(
                f"test metrics: summary not written ({type(exc).__name__}: {exc}); "
                f"rerun `python tests/perf_report.py summarize {self.root}`\n"
            )


# ------------------------------------------------------------------- reading


def read_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Records of one events file, plus the count of unreadable lines (a process
    killed mid-write can leave a truncated last line)."""
    records: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            malformed += 1
    return records, malformed


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        return {"_unreadable": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"_unreadable": "not a JSON object"}


class _Process:
    def __init__(self, filename: str, records: list[dict[str, Any]], malformed: int) -> None:
        self.filename = filename
        self.malformed = malformed
        self.name = filename
        self.role = "unknown"
        self.pid: int | None = None
        self.first: dict[str, dict[str, Any]] = {}
        self.last_totals: dict[str, Any] | None = None
        self.collection: dict[str, Any] | None = None
        self.started: dict[str, float] = {}
        self.phases: list[dict[str, Any]] = []
        self.finished: set[str] = set()
        self.spawns: list[dict[str, Any]] = []
        self.node_down: list[dict[str, Any]] = []
        self.last_mono: float | None = None
        for record in records:
            kind = str(record.get("kind"))
            mono = record.get("mono")
            if isinstance(mono, (int, float)):
                self.last_mono = mono if self.last_mono is None else max(self.last_mono, mono)
            self.first.setdefault(kind, record)
            if kind == "session_start":
                self.name = str(record.get("worker", filename))
                self.role = str(record.get("role", "unknown"))
                self.pid = record.get("pid")
            elif kind == "start":
                self.started[str(record.get("nodeid"))] = float(record.get("mono", 0.0))
            elif kind == "phase":
                self.phases.append(record)
                if record.get("when") == "teardown":
                    self.finished.add(str(record.get("nodeid")))
            elif kind == "totals":
                self.last_totals = record
            elif kind == "collection":
                self.collection = record
            elif kind == "spawns":
                self.spawns.append(record)
            elif kind == "node_down":
                self.node_down.append(record)

    def mono(self, kind: str) -> float | None:
        value = self.first.get(kind, {}).get("mono")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def unfinished(self) -> list[str]:
        return [nodeid for nodeid in self.started if nodeid not in self.finished]


def _test_outcome(phases: list[dict[str, Any]]) -> str:
    for phase in phases:
        if phase.get("outcome") == "failed":
            return "failed" if phase.get("when") == "call" else "error"
    for phase in phases:
        if phase.get("outcome") == "skipped":
            return "xfailed" if phase.get("xfail") else "skipped"
    for phase in phases:
        if phase.get("xfail"):
            return "xpassed"
    return "passed"


def _seconds(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _span(start: float | None, end: float | None) -> float | None:
    return None if start is None or end is None else round(end - start, 3)


def _status(main: _Process | None, runner: dict[str, Any] | None) -> tuple[str, str]:
    state = (runner or {}).get("state")
    finish = main.first.get("session_finish") if main else None
    if state == "spawn_failed":
        return "spawn_failed", "the wrapper could not start pytest"
    if state == "timed_out":
        return "timed_out", "the wrapper's deadline expired and it killed the pytest process tree"
    if state == "interrupted":
        return (
            "interrupted",
            "the wrapper was interrupted or terminated and stopped the pytest process tree",
        )
    if finish is None:
        code = (runner or {}).get("exit_code")
        if state == "exited" and isinstance(code, int) and main is None:
            name = EXIT_STATUS_NAMES.get(code, f"exit_{code}")
            return name, f"pytest exited {code} before its session started (e.g. a usage error)"
        if state == "running":
            return "killed_externally", (
                "neither pytest nor the wrapper finished: both were killed from outside "
                "(a CI step timeout, job cancellation, or a crash); records end at the last flushed line"
            )
        return "unfinished", (
            "pytest recorded no session_finish: it was killed, cancelled or crashed before "
            "finishing; records end at the last flushed line"
        )
    exitstatus = finish.get("exitstatus")
    name = (
        EXIT_STATUS_NAMES.get(exitstatus, f"exit_{exitstatus}")
        if isinstance(exitstatus, int)
        else "unknown"
    )
    return name, f"pytest finished with exit status {exitstatus}"


def summarize(root: Path) -> dict[str, Any]:
    """Derive the run's summary from the files in ``root``. Works on any prefix of
    a run: a killed run is summarized from whatever records reached the disk."""
    manifest = _read_json(root / MANIFEST)
    runner = _read_json(root / RUNNER)
    processes: list[_Process] = []
    for path in sorted(root.glob(f"{EVENTS_PREFIX}*.jsonl")):
        records, malformed = read_events(path)
        processes.append(_Process(path.name, records, malformed))
    notes: list[str] = []
    mains = [p for p in processes if p.role in ("controller", "main")]
    main = mains[0] if mains else None
    if len(mains) > 1:
        notes.append(
            f"{len(mains)} controlling processes recorded; using {main.filename if main else '?'}"
        )
    if main is None:
        notes.append("no controlling process recorded a session_start")
    workers = [p for p in processes if p.role in ("worker", "main")]

    # Worker-seconds: summed phase durations. Never compared to elapsed below.
    by_phase: Counter[str] = Counter()
    by_file: dict[str, list[float]] = {}
    per_test: dict[str, list[dict[str, Any]]] = {}
    test_worker: dict[str, str] = {}
    worker_rows: dict[str, dict[str, Any]] = {}
    last_phase_mono: float | None = None
    for proc in workers:
        busy = 0.0
        for phase in proc.phases:
            duration = float(phase.get("duration") or 0.0)
            busy += duration
            by_phase[str(phase.get("when"))] += duration
            nodeid = str(phase.get("nodeid"))
            per_test.setdefault(nodeid, []).append(phase)
            test_worker[nodeid] = proc.name
            mono = phase.get("mono")
            if isinstance(mono, (int, float)):
                last_phase_mono = mono if last_phase_mono is None else max(last_phase_mono, mono)
        first_start = min(proc.started.values()) if proc.started else None
        unfinished = proc.unfinished
        worker_rows[proc.name] = {
            "pid": proc.pid,
            "worker_seconds": round(busy, 3),
            "tests_started": len(proc.started),
            "tests_finished": len(proc.finished),
            "span_seconds": _span(first_start, proc.last_mono),
            "collection_seconds": (proc.collection or {}).get("seconds"),
            "last_active": unfinished[-1] if unfinished else None,
            "finished_session": "session_finish" in proc.first,
            "malformed_lines": proc.malformed,
        }
    for nodeid, phases in per_test.items():
        slot = by_file.setdefault(nodeid.split("::", 1)[0], [0.0, 0])
        slot[0] += sum(float(p.get("duration") or 0.0) for p in phases)
        slot[1] += 1

    finished_tests = {nodeid for proc in workers for nodeid in proc.finished}
    # A test with no teardown report has no verdict yet, whatever its setup said.
    outcomes: Counter[str] = Counter(
        _test_outcome(phases) if nodeid in finished_tests else "unfinished"
        for nodeid, phases in per_test.items()
    )
    started = {nodeid for proc in workers for nodeid in proc.started}
    unfinished_tests = [
        {"nodeid": nodeid, "worker": proc.name} for proc in workers for nodeid in proc.unfinished
    ]

    inventories = []
    for path in sorted(root.glob(f"{INVENTORY_PREFIX}*.json")):
        data = _read_json(path)
        if data and "_unreadable" not in data:
            inventories.append(data)
        else:
            notes.append(f"{path.name} unreadable")
    inventory: dict[str, Any] = {
        "collected": None,
        "selected": None,
        "deselected": None,
        "consistent": None,
    }
    selected_set: set[str] | None = None
    if inventories:
        full = next((inv for inv in inventories if "selected" in inv), None)
        reference = full or inventories[0]
        counts = reference.get("counts") or {}
        digests = [inv.get("digests") for inv in inventories]
        inventory = {
            **{key: counts.get(key) for key in ("collected", "selected", "deselected")},
            "collect_errors": reference.get("collect_errors"),
            "consistent": all(d == digests[0] for d in digests),
            "files": len(inventories),
        }
        if full is None:
            notes.append("no full nodeid inventory (main/gw0) was written; not_started is unknown")
        else:
            selected_set = set(full["selected"])

    fixtures: dict[tuple[str, str], list[float]] = {}
    spawn_commands: Counter[str] = Counter()
    spawn_sites: Counter[str] = Counter()
    spawn_errors = 0
    spawn_mode = False
    for proc in processes:
        totals = proc.last_totals or {}
        for name, scope, count, seconds in totals.get("fixtures") or []:
            slot = fixtures.setdefault((str(name), str(scope)), [0, 0.0])
            slot[0] += count
            slot[1] += seconds
        spawns = totals.get("spawns")
        if isinstance(spawns, dict):
            spawn_mode = True
            spawn_commands.update(spawns.get("by_command") or {})
            spawn_sites.update(spawns.get("by_site") or {})
            spawn_errors += int(spawns.get("hook_errors") or 0)
    spawn_tests: Counter[str] = Counter()
    for proc in workers:
        for record in proc.spawns:
            spawn_tests[str(record.get("nodeid"))] += int(record.get("count") or 0)

    status, detail = _status(main, runner)
    main_start = main.mono("session_start") if main else None
    main_finish = main.mono("session_finish") if main else None
    observed_end = max((p.last_mono for p in processes if p.last_mono is not None), default=None)
    runner_end = (runner or {}).get("end_mono")
    runner_end = float(runner_end) if isinstance(runner_end, (int, float)) else None
    finish_record = main.first.get("session_finish", {}) if main else {}

    summary: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "run_id": (manifest or {}).get("run_id"),
        "status": status,
        "status_detail": detail,
        "exitstatus": finish_record.get("exitstatus"),
        "runner_exit_code": (runner or {}).get("exit_code"),
        "manifest": manifest,
        "elapsed": {
            "note": "wall-clock time on one clock; never a sum across workers",
            "session_seconds": _span(main_start, main_finish),
            "observed_span_seconds": _span(main_start, observed_end),
            "process_seconds": (runner or {}).get("elapsed_seconds"),
            "cleanup_tail": {
                "last_report_to_sessionfinish": _span(
                    last_phase_mono, main.mono("sessionfinish_start") if main else None
                ),
                "sessionfinish": _span(
                    main.mono("sessionfinish_start") if main else None, main_finish
                ),
                "sessionfinish_to_unconfigure": _span(
                    main_finish, main.mono("unconfigure") if main else None
                ),
                "unconfigure_to_exit": _span(
                    main.mono("unconfigure") if main else None, main.mono("exit") if main else None
                ),
                "exit_to_process_exit": _span(main.mono("exit") if main else None, runner_end),
                "last_report_to_process_exit": _span(last_phase_mono, runner_end),
            },
        },
        "worker_seconds": {
            "note": "test phase durations summed across workers; exceeds elapsed when workers overlap",
            "total": round(sum(by_phase.values()), 3),
            "by_phase": {k: round(v, 3) for k, v in sorted(by_phase.items())},
            "by_worker": worker_rows,
        },
        "tests": {
            **inventory,
            "started": len(started),
            "finished": sum(len(p.finished) for p in workers),
            "unfinished": unfinished_tests,
            "not_started": len(selected_set - started) if selected_set is not None else None,
            "outcomes": dict(sorted(outcomes.items())),
        },
        "files": [
            {"file": name, "worker_seconds": round(slot[0], 3), "tests": int(slot[1])}
            for name, slot in sorted(by_file.items(), key=lambda kv: -kv[1][0])[:TOP_N]
        ],
        "slowest_tests": [
            {
                "nodeid": nodeid,
                "worker_seconds": round(sum(float(p.get("duration") or 0.0) for p in phases), 3),
                "worker": test_worker.get(nodeid),
            }
            for nodeid, phases in sorted(
                per_test.items(),
                key=lambda kv: -sum(float(p.get("duration") or 0.0) for p in kv[1]),
            )[:TOP_N]
        ],
        "fixtures": [
            {
                "fixture": name,
                "scope": scope,
                "setups": int(slot[0]),
                "worker_seconds": round(slot[1], 3),
            }
            for (name, scope), slot in sorted(fixtures.items(), key=lambda kv: -kv[1][1])[:TOP_N]
        ],
        "subprocesses": (
            {
                "total": sum(spawn_commands.values()),
                "by_command": dict(spawn_commands.most_common(TOP_N)),
                "by_site": dict(spawn_sites.most_common(TOP_N)),
                "by_test": dict(spawn_tests.most_common(TOP_N)),
                "hook_errors": spawn_errors,
            }
            if spawn_mode
            else None
        ),
        "node_down": [r for p in processes for r in p.node_down],
        "malformed_lines": sum(p.malformed for p in processes),
        "notes": notes,
    }
    return summary


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def render_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    elapsed = summary["elapsed"]
    ws = summary["worker_seconds"]
    tests = summary["tests"]
    lines.append(
        f"test metrics run {summary.get('run_id')}: {summary['status']} -- {summary['status_detail']}"
    )
    lines.append(
        "elapsed (wall clock): session "
        f"{_fmt(elapsed['session_seconds'])} | observed span {_fmt(elapsed['observed_span_seconds'])} | "
        f"pytest process lifetime {_fmt(elapsed['process_seconds'])}"
    )
    phases = ", ".join(f"{k} {v:.1f}" for k, v in ws["by_phase"].items())
    lines.append(
        f"worker-seconds (summed across {len(ws['by_worker'])} worker(s); not wall time): "
        f"{ws['total']:.1f} ({phases})"
    )
    lines.append(
        f"tests: collected {tests.get('collected')} | selected {tests.get('selected')} | "
        f"deselected {tests.get('deselected')} | started {tests['started']} | finished {tests['finished']} | "
        f"not started {tests['not_started']} | unfinished {len(tests['unfinished'])}"
    )
    lines.append(
        "outcomes: " + (", ".join(f"{k} {v}" for k, v in tests["outcomes"].items()) or "none")
    )
    tail = elapsed["cleanup_tail"]
    lines.append(
        "cleanup tail: " + ", ".join(f"{k.replace('_', ' ')} {_fmt(v)}" for k, v in tail.items())
    )
    lines.append("workers:")
    for name, row in sorted(ws["by_worker"].items()):
        lines.append(
            f"  {name}: {row['worker_seconds']:.1f} worker-s, {row['tests_finished']}/{row['tests_started']} "
            f"tests finished, span {_fmt(row['span_seconds'])}, collection {_fmt(row['collection_seconds'])}"
            + ("" if row["finished_session"] else ", NO session_finish")
        )
    if tests["unfinished"]:
        lines.append("unfinished tests (started, no teardown report):")
        lines.extend(f"  {row['worker']}: {row['nodeid']}" for row in tests["unfinished"])
    lines.append("top files by worker-seconds:")
    lines.extend(
        f"  {row['worker_seconds']:9.1f}  {row['tests']:6d}  {row['file']}"
        for row in summary["files"][:15]
    )
    lines.append("top fixtures by setup worker-seconds:")
    lines.extend(
        f"  {row['worker_seconds']:9.1f}  {row['setups']:6d}  {row['fixture']} ({row['scope']})"
        for row in summary["fixtures"][:10]
    )
    lines.append("slowest tests (worker-seconds):")
    lines.extend(
        f"  {row['worker_seconds']:9.1f}  {row['nodeid']}" for row in summary["slowest_tests"][:10]
    )
    spawns = summary.get("subprocesses")
    if spawns:
        lines.append(
            f"subprocess launches: {spawns['total']} (hook errors {spawns['hook_errors']})"
        )
        lines.extend(
            f"  {count:8d}  {name}" for name, count in list(spawns["by_command"].items())[:15]
        )
    if summary["node_down"]:
        lines.append(
            "xdist workers down: "
            + ", ".join(
                f"{r.get('worker')}{' (crashed)' if r.get('crashed') else ''}"
                for r in summary["node_down"]
            )
        )
    if summary["malformed_lines"]:
        lines.append(f"unreadable event lines: {summary['malformed_lines']}")
    lines.extend(f"note: {note}" for note in summary["notes"])
    return "\n".join(lines) + "\n"


def write_summary(root: Path) -> dict[str, Any]:
    summary = summarize(root)
    _write_json_atomic(root / SUMMARY_JSON, summary)
    tmp = root / f".{SUMMARY_TEXT}.{os.getpid()}.tmp"
    tmp.write_text(render_text(summary), encoding="utf-8")
    os.replace(tmp, root / SUMMARY_TEXT)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Summarize a --test-metrics-dir directory.")
    sub = parser.add_subparsers(dest="command", required=True)
    summarize_cmd = sub.add_parser(
        "summarize", help="(re)write summary.json/summary.txt and print the text"
    )
    summarize_cmd.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    summary = write_summary(args.directory)
    sys.stdout.write(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
