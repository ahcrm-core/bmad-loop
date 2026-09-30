"""Harness tests for the opt-in runtime metrics (tests/perf_report.py) and the
pytest-lifetime wrapper (scripts/run_test_benchmark.py).

Behavior that needs a real pytest session runs a small synthetic suite in a child
pytest process: a temp directory with its own ``pytest.ini`` (so the repository's
``addopts`` and conftest stay out of it) and a conftest wired exactly like
``tests/conftest.py``. Summary arithmetic and partial-file reading are tested on
hand-written event files instead, which is cheaper and states the numbers.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import perf_report
import pytest

from bmad_loop.process_host import get_process_host

TESTS_DIR = Path(__file__).resolve().parent
WRAPPER = TESTS_DIR.parent / "scripts" / "run_test_benchmark.py"
sys.path.insert(0, str(WRAPPER.parent))
import run_test_benchmark  # noqa: E402

# Unique per session: a match anywhere in a metrics file can only be a leak.
CANARY_ENV = f"canary-env-{uuid.uuid4().hex}"
CANARY_OUT = f"canary-out-{uuid.uuid4().hex}"
CANARY_ARGV = f"canary-argv-{uuid.uuid4().hex}"
CANARY_FAIL = f"canary-fail-{uuid.uuid4().hex}"

SYNTH_CONFTEST = f"""\
import sys
sys.path.insert(0, {str(TESTS_DIR)!r})
import perf_report


def pytest_addoption(parser):
    perf_report.add_options(parser)


def pytest_configure(config):
    perf_report.configure(config)
"""

# Bound on any one synthetic child run; a hang fails the test instead of the job.
CHILD_TIMEOUT_S = 180


def _child_env(**extra: str) -> dict[str, str]:
    # The outer run's PYTEST_* variables (xdist worker identity, addopts) must not
    # leak into the synthetic session.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env.update(extra)
    return env


def _suite(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\naddopts = -p no:cacheprovider\n", encoding="utf-8")
    (root / "conftest.py").write_text(SYNTH_CONFTEST, encoding="utf-8")
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8")
    return root


def _run(argv: list[str], cwd: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # fixed argv, no shell
        argv,
        cwd=cwd,
        env=_child_env(**env),
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        check=False,
    )


def _pytest(suite: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, "-m", "pytest", *args], suite, **env)


def _wrapped(
    suite: Path, metrics: Path, *pytest_args: str, wrapper_args: tuple[str, ...] = (), **env: str
) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable,
        str(WRAPPER),
        "--metrics-dir",
        str(metrics),
        *wrapper_args,
        "--",
        *pytest_args,
    ]
    return _run(argv, suite, **env)


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return f"rc={proc.returncode}\n--- stdout\n{proc.stdout[-4000:]}\n--- stderr\n{proc.stderr[-4000:]}"


def _events(metrics: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for path in sorted(metrics.glob("events-*.jsonl")):
        records, malformed = perf_report.read_events(path)
        assert malformed == 0, path
        out[path.name] = records
    return out


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------ a real xdist run, shared read-only

MIXED_SUITE = """\
import os
import subprocess
import sys

import pytest


def test_passes():
    pass


def test_prints_and_spawns():
    print(os.environ["METRICS_CANARY_OUT"])
    print(os.environ["METRICS_CANARY_OUT"], file=sys.stderr)
    subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", os.environ["METRICS_CANARY_ARGV"]],
        check=True,
    )
    subprocess.run(["git", "-c", "metrics.canary=" + os.environ["METRICS_CANARY_ARGV"], "version"], check=True)


def test_fails():
    assert os.environ["METRICS_CANARY_FAIL"] == "", "fails on purpose"


@pytest.mark.skip(reason="static skip")
def test_skipped():
    pass


def test_deselected_beta():
    pass


@pytest.mark.xdist_group("grouped")
def test_grouped():
    pass


@pytest.mark.parametrize("n", range(6))
def test_many(n):
    pass
"""


@pytest.fixture(scope="module")
def xdist_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """One two-worker loadgroup run through the wrapper, with a failure, a skip, a
    deselection, a grouped test, printed output and subprocess attribution on."""
    base = tmp_path_factory.mktemp("xdist-run")
    suite = _suite(base / "suite", {"test_mixed.py": MIXED_SUITE})
    metrics = base / "metrics"
    proc = _wrapped(
        suite,
        metrics,
        "-q",
        "-rP",  # print passed tests' captured output, so the canary provably reaches pytest
        "-n",
        "2",
        "--dist",
        "loadgroup",
        "-k",
        "not deselected_beta",
        "--test-metrics-subprocesses",
        METRICS_CANARY_ENV=CANARY_ENV,
        METRICS_CANARY_OUT=CANARY_OUT,
        METRICS_CANARY_ARGV=CANARY_ARGV,
        METRICS_CANARY_FAIL=CANARY_FAIL,
    )
    return proc, metrics, suite


def test_wrapper_preserves_a_failed_runs_exit_status(xdist_run):
    proc, metrics, _ = xdist_run
    assert proc.returncode == 1, _output(proc)
    runner = _json(metrics / perf_report.RUNNER)
    assert (runner["state"], runner["exit_code"]) == ("exited", 1)
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    assert (summary["status"], summary["exitstatus"], summary["runner_exit_code"]) == (
        "tests_failed",
        1,
        1,
    )
    assert summary["tests"]["outcomes"] == {"failed": 1, "passed": 9, "skipped": 1}


def test_worker_files_do_not_collide(xdist_run):
    proc, metrics, _ = xdist_run
    events = _events(metrics)
    starts = {name: records[0] for name, records in events.items()}
    assert all(record["kind"] == "session_start" for record in starts.values())
    # Each file is named for, and only written by, the process that opened it.
    for name, record in starts.items():
        assert name == f"events-{record['worker']}-{record['pid']}.jsonl"
    roles = sorted(record["role"] for record in starts.values())
    assert roles == ["controller", "worker", "worker"], _output(proc)
    assert len({(r["worker"], r["pid"]) for r in starts.values()}) == len(starts)

    # Every selected test ran on exactly one worker; no record is duplicated.
    executed: list[str] = []
    for records in events.values():
        executed.extend(r["nodeid"] for r in records if r["kind"] == "start")
    selected = _json(next(metrics.glob("inventory-gw0-*.json")))["selected"]
    assert sorted(executed) == sorted(selected)
    assert len(executed) == len(set(executed)) == 11


def test_collected_and_selected_inventories_are_distinct(xdist_run):
    _, metrics, _ = xdist_run
    inventories = {
        path.name.split("-")[1]: _json(path) for path in metrics.glob("inventory-*.json")
    }
    assert sorted(inventories) == ["gw0", "gw1"]  # both workers collect; the controller does not
    first = inventories["gw0"]
    # One full copy; the other worker's digests must match it.
    assert "selected" not in inventories["gw1"]
    assert inventories["gw1"]["digests"] == first["digests"]
    assert first["digests"]["selected"] == perf_report.nodeid_digest(first["selected"])
    assert first["digests"]["collected"] != first["digests"]["selected"]
    beta = "test_mixed.py::test_deselected_beta"
    assert beta in first["collected"]
    assert beta not in first["selected"]
    assert first["deselected"] == [beta]
    assert len(first["collected"]) == len(first["selected"]) + 1
    # loadgroup's worker-only `@group` suffix is stripped, so inventories compare
    # equal with and without -n.
    assert "test_mixed.py::test_grouped" in first["selected"]
    assert not any("@" in nodeid for nodeid in first["collected"])
    summary = _json(metrics / perf_report.SUMMARY_JSON)["tests"]
    assert (summary["collected"], summary["selected"], summary["deselected"]) == (12, 11, 1)
    assert summary["consistent"] is True
    assert (summary["started"], summary["not_started"], summary["unfinished"]) == (11, 0, [])


def test_no_raw_environment_or_child_output_is_recorded(xdist_run):
    proc, metrics, suite = xdist_run
    # The canaries did reach the child's output and the run's subprocesses ...
    assert CANARY_OUT in proc.stdout + proc.stderr, _output(proc)
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    spawns = summary["subprocesses"]
    assert spawns["by_command"].get("git version") == 1
    assert spawns["by_command"].get("python", 0) >= 1  # the controller also launches its workers
    assert spawns["by_test"] == {"test_mixed.py::test_prints_and_spawns": 2}
    # ... and none of them, nor the machine-specific suite path, reached a file.
    files = sorted(p for p in metrics.iterdir() if p.is_file())
    assert {p.name for p in files} >= {
        perf_report.MANIFEST,
        perf_report.RUNNER,
        perf_report.SUMMARY_JSON,
    }
    for path in files:
        text = path.read_text(encoding="utf-8")
        for secret in (CANARY_ENV, CANARY_OUT, CANARY_ARGV, CANARY_FAIL, str(suite)):
            assert secret not in text, f"{secret!r} leaked into {path.name}"


def test_summary_reports_both_clocks_for_a_real_run(xdist_run):
    _, metrics, _ = xdist_run
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    assert summary["elapsed"]["session_seconds"] > 0
    assert summary["elapsed"]["process_seconds"] >= summary["elapsed"]["session_seconds"]
    assert set(summary["worker_seconds"]["by_worker"]) == {"gw0", "gw1"}
    manifest = summary["manifest"]
    assert (manifest["numprocesses"], manifest["dist"], manifest["subprocess_attribution"]) == (
        2,
        "loadgroup",
        True,
    )
    assert manifest["selection"] == {
        "args": ["."],
        "ignore": [],
        "keyword_expression": True,
        "mark_expression": False,
    }


# ------------------------------------------------------------- the default path


INERT_SUITE = """\
import json
import os
from pathlib import Path


def test_reports_registration(pytestconfig):
    Path(os.environ["METRICS_PROBE_OUT"]).write_text(json.dumps({
        "registered": pytestconfig.pluginmanager.get_plugin("bmad-loop-test-metrics") is not None,
        "option": pytestconfig.getoption("test_metrics_dir"),
    }), encoding="utf-8")
"""


def test_without_the_option_nothing_is_registered_or_written(tmp_path):
    suite = _suite(tmp_path / "suite", {"test_inert.py": INERT_SUITE})
    probe = tmp_path / "probe.json"
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}

    plain = _pytest(suite, "-q", METRICS_PROBE_OUT=str(probe))
    assert plain.returncode == 0, _output(plain)
    assert _json(probe) == {"registered": False, "option": None}
    created = (
        {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
        - before
        - {probe.relative_to(tmp_path)}
    )
    assert {p for p in created if "__pycache__" not in p.parts} == set()

    # The same test sees the plugin when the option is given (through the
    # wrapper, whose exit status is the passing run's 0) — so the check above
    # can tell the two apart.
    wrapped = _wrapped(suite, tmp_path / "metrics", "-q", METRICS_PROBE_OUT=str(probe))
    assert wrapped.returncode == 0, _output(wrapped)
    assert _json(probe)["registered"] is True
    assert _json(tmp_path / "metrics" / perf_report.RUNNER)["exit_code"] == 0
    assert _json(tmp_path / "metrics" / perf_report.SUMMARY_JSON)["status"] == "passed"


def test_this_suite_registers_the_options(pytestconfig):
    # Raises ValueError if tests/conftest.py stopped registering it.
    pytestconfig.getoption("test_metrics_dir")
    pytestconfig.getoption("test_metrics_subprocesses")


def test_usage_errors_keep_pytest_exit_status(tmp_path):
    suite = _suite(tmp_path / "suite", {"test_inert.py": "def test_ok():\n    pass\n"})

    orphan_flag = _pytest(suite, "-q", "--test-metrics-subprocesses")
    assert orphan_flag.returncode == 4, _output(orphan_flag)
    assert "requires --test-metrics-dir" in orphan_flag.stderr

    used = tmp_path / "used"
    used.mkdir()
    (used / perf_report.MANIFEST).write_text("{}", encoding="utf-8")
    reuse = _pytest(suite, "-q", f"--test-metrics-dir={used}")
    assert reuse.returncode == 4, _output(reuse)
    assert "already holds a metrics run" in reuse.stderr
    assert sorted(p.name for p in used.iterdir()) == [perf_report.MANIFEST]

    bad_arg = _wrapped(suite, tmp_path / "metrics", "-q", "--no-such-pytest-flag")
    assert bad_arg.returncode == 4, _output(bad_arg)
    runner = _json(tmp_path / "metrics" / perf_report.RUNNER)
    assert (runner["state"], runner["exit_code"]) == ("exited", 4)
    assert _json(tmp_path / "metrics" / perf_report.SUMMARY_JSON)["status"] == "usage_error"


# ------------------------------------------------------------ summary arithmetic


def _write_events(path: Path, records: list[dict[str, Any]], tail: str = "") -> None:
    lines = [json.dumps({"v": 1, **r}) for r in records]
    path.write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")


def _worker_records(name: str, pid: int, tests: list[tuple[str, float]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {"kind": "session_start", "mono": 100.0, "worker": name, "role": "worker", "pid": pid}
    ]
    mono = 100.0
    for nodeid, seconds in tests:
        records.append({"kind": "start", "mono": mono, "nodeid": nodeid})
        for when, duration in (("setup", 0.0), ("call", seconds), ("teardown", 0.0)):
            mono += duration
            records.append(
                {
                    "kind": "phase",
                    "mono": mono,
                    "nodeid": nodeid,
                    "when": when,
                    "outcome": "passed",
                    "duration": duration,
                }
            )
    return records


def test_summary_does_not_conflate_worker_seconds_with_elapsed(tmp_path):
    # Two workers, each busy for 2s over the same 2s of wall time.
    _write_events(
        tmp_path / "events-controller-1.jsonl",
        [
            {
                "kind": "session_start",
                "mono": 100.0,
                "worker": "controller",
                "role": "controller",
                "pid": 1,
            },
            {"kind": "sessionfinish_start", "mono": 102.0, "exitstatus": 0},
            {"kind": "session_finish", "mono": 102.0, "exitstatus": 0},
        ],
    )
    _write_events(
        tmp_path / "events-gw0-2.jsonl",
        _worker_records("gw0", 2, [("a.py::t1", 1.0), ("a.py::t2", 1.0)]),
    )
    _write_events(
        tmp_path / "events-gw1-3.jsonl",
        _worker_records("gw1", 3, [("b.py::t3", 1.0), ("b.py::t4", 1.0)]),
    )

    summary = perf_report.summarize(tmp_path)
    assert summary["worker_seconds"]["total"] == 4.0
    assert summary["elapsed"]["session_seconds"] == 2.0
    assert {row["worker_seconds"] for row in summary["worker_seconds"]["by_worker"].values()} == {
        2.0
    }
    assert summary["files"] == [
        {"file": "a.py", "worker_seconds": 2.0, "tests": 2},
        {"file": "b.py", "worker_seconds": 2.0, "tests": 2},
    ]
    text = perf_report.render_text(summary)
    assert "elapsed (wall clock): session 2.0s" in text
    assert "worker-seconds (summed across 2 worker(s); not wall time): 4.0" in text


def test_a_killed_run_is_summarized_from_its_flushed_records(tmp_path):
    (tmp_path / "inventory-gw0-2.json").write_text(
        json.dumps(
            {
                "collected": ["a.py::t1", "a.py::t2", "a.py::t3"],
                "selected": ["a.py::t1", "a.py::t2", "a.py::t3"],
                "deselected": [],
                "counts": {"collected": 3, "selected": 3, "deselected": 0},
            }
        ),
        encoding="utf-8",
    )
    _write_events(
        tmp_path / "events-controller-1.jsonl",
        [
            {
                "kind": "session_start",
                "mono": 100.0,
                "worker": "controller",
                "role": "controller",
                "pid": 1,
            }
        ],
    )
    worker = _worker_records("gw0", 2, [("a.py::t1", 1.0)])
    worker.append({"kind": "start", "mono": 101.0, "nodeid": "a.py::t2"})
    # Killed mid-line: the truncated record is counted, not fatal.
    _write_events(tmp_path / "events-gw0-2.jsonl", worker, tail='{"v":1,"kind":"pha')

    summary = perf_report.summarize(tmp_path)
    assert summary["status"] == "unfinished"
    assert summary["malformed_lines"] == 1
    assert summary["tests"]["unfinished"] == [{"nodeid": "a.py::t2", "worker": "gw0"}]
    assert (
        summary["tests"]["started"],
        summary["tests"]["finished"],
        summary["tests"]["not_started"],
    ) == (2, 1, 1)
    assert summary["worker_seconds"]["by_worker"]["gw0"]["last_active"] == "a.py::t2"
    assert summary["elapsed"]["session_seconds"] is None
    assert summary["elapsed"]["observed_span_seconds"] == 1.0

    # The wrapper having been killed too (runner still "running") reads as an
    # external kill — a CI step timeout or cancellation — not a pytest verdict.
    (tmp_path / perf_report.RUNNER).write_text(json.dumps({"state": "running"}), encoding="utf-8")
    assert perf_report.summarize(tmp_path)["status"] == "killed_externally"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["git", "-C", "/secret/repo", "-c", "user.name=x", "--no-pager", "rev-parse", "HEAD"],
            "git rev-parse",
        ),
        (["C:\\Program Files\\Git\\cmd\\git.exe", "ls-files", "-z"], "git ls-files"),
        (["git", "--version"], "git <no-verb>"),
        (["git", "/tmp/not-a-verb"], "git <other>"),
        (["/usr/bin/python3.13", "-c", "print(1)"], "python"),
        ('"C:\\Program Files\\Python\\pythonw.exe" -c pass', "python"),
        ("tmux new-session -d", "tmux"),
        (["/tmp/some dir/odd name!"], "<other>"),
    ],
)
def test_commands_are_reduced_to_program_and_git_verb(args, expected):
    assert perf_report.describe_command(None, args) == expected


# -------------------------------------------------------------- the wrapper


def _gone(pid: int) -> bool:
    """Dead, or a zombie waiting for its (new) parent to reap it."""
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            return True
        return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")
    return not get_process_host().is_alive(pid)


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _gone(pid):
            return True
        time.sleep(0.05)
    return _gone(pid)


@pytest.fixture
def reap_leftovers() -> Iterator[list[int]]:
    """Pids a failing row may leak; force-killed at teardown so a red row never
    leaves a 120s sleeper behind."""
    pids: list[int] = []
    yield pids
    host = get_process_host()
    for pid in pids:
        if not _gone(pid):
            with contextlib.suppress(Exception):
                host.force_kill(pid)


SLEEPER_SUITE = """\
import os
import subprocess
import sys
import time
from pathlib import Path


def test_first():
    pass


def test_sleeper():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(os.environ["METRICS_PIDS"]).write_text(f"{os.getpid()} {child.pid}", encoding="utf-8")
    time.sleep(120)
"""

WRAPPER_DEADLINE_S = 15.0
# POSIX also gets the graceful path (SIGINT, then SIGKILL after the grace). On
# Windows pytest shares the wrapper's console group, so no Ctrl+C can be aimed
# at it alone: the deadline kills the tree at once and the grace does not apply.
GRACE_ROWS = [0.0] if sys.platform == "win32" else [0.0, 10.0]


@pytest.mark.parametrize("grace", GRACE_ROWS)
def test_wrapper_deadline_stops_the_tree_and_keeps_partial_records(tmp_path, reap_leftovers, grace):
    suite = _suite(tmp_path / "suite", {"test_sleep.py": SLEEPER_SUITE})
    metrics = tmp_path / "metrics"
    pids_file = tmp_path / "pids"
    started = time.monotonic()
    proc = _wrapped(
        suite,
        metrics,
        "-q",
        wrapper_args=("--timeout", str(WRAPPER_DEADLINE_S), "--grace", str(grace)),
        METRICS_PIDS=str(pids_file),
    )
    elapsed = time.monotonic() - started
    assert pids_file.exists(), f"the sleeper never started before the deadline\n{_output(proc)}"
    pids = [int(token) for token in pids_file.read_text(encoding="utf-8").split()]
    reap_leftovers.extend(pids)

    assert proc.returncode == run_test_benchmark.EXIT_TIMED_OUT, _output(proc)
    # Bounded: the deadline, the grace, the wrapper's own reap wait, and startup.
    assert elapsed < WRAPPER_DEADLINE_S + grace + 30, elapsed
    for pid in pids:
        assert _wait_gone(pid), f"pid {pid} outlived the wrapper's tree kill"

    runner = _json(metrics / perf_report.RUNNER)
    assert runner["state"] == "timed_out"
    assert runner["child_reaped"] is True
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    assert summary["status"] == "timed_out"
    assert summary["tests"]["unfinished"] == [
        {"nodeid": "test_sleep.py::test_sleeper", "worker": "main"}
    ]
    assert summary["tests"]["outcomes"] == {"passed": 1, "unfinished": 1}
    finish = [
        r for records in _events(metrics).values() for r in records if r["kind"] == "session_finish"
    ]
    if grace:
        # SIGINT first: pytest still closed its session, as interrupted.
        assert [r["exitstatus"] for r in finish] == [2]
    else:
        assert finish == []


def test_wrapper_reports_a_spawn_failure(tmp_path, monkeypatch):
    def no_spawn(*_args, **_kwargs):
        raise FileNotFoundError("no interpreter")

    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", no_spawn)
    rc = run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--", "-q"])
    assert rc == run_test_benchmark.EXIT_SPAWN_FAILED
    runner = _json(tmp_path / "m" / perf_report.RUNNER)
    assert (runner["state"], runner["error"], runner["exit_code"]) == (
        "spawn_failed",
        "FileNotFoundError",
        None,
    )
    assert _json(tmp_path / "m" / perf_report.SUMMARY_JSON)["status"] == "spawn_failed"


def test_wrapper_refuses_a_used_directory_and_a_passed_deadline(tmp_path, monkeypatch):
    def no_spawn(*_args, **_kwargs):
        raise AssertionError("must not start pytest")

    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", no_spawn)
    used = tmp_path / "used"
    used.mkdir()
    (used / perf_report.RUNNER).write_text("{}", encoding="utf-8")
    assert (
        run_test_benchmark.main(["--metrics-dir", str(used), "--", "-q"])
        == run_test_benchmark.EXIT_USAGE
    )

    late = tmp_path / "late"
    rc = run_test_benchmark.main(
        ["--metrics-dir", str(late), "--deadline-epoch", str(time.time() - 1), "--", "-q"]
    )
    assert rc == run_test_benchmark.EXIT_TIMED_OUT
    runner = _json(late / perf_report.RUNNER)
    assert (runner["state"], runner["child_started"]) == ("timed_out", False)
