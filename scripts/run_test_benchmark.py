#!/usr/bin/env python3
"""Run pytest with ``--test-metrics-dir`` and record the pytest process lifetime.

    uv run python scripts/run_test_benchmark.py --metrics-dir DIR [--timeout S |
        --deadline-epoch T] [--grace S] -- <pytest arguments>

The in-process metrics (tests/perf_report.py) cannot see the start of the
interpreter, or anything after its own ``atexit`` record: import time, the tail of
temp-directory cleanup, and a process that never exits at all. This wrapper owns
that span. It writes ``runner.json`` into the metrics directory before starting
pytest (``state: running``) and rewrites it when pytest ends, then regenerates
``summary.json``/``summary.txt`` and prints the text summary.

``runner.json`` states: ``running`` (the wrapper itself was killed, e.g. by a CI
step timeout or job cancellation), ``exited``, ``timed_out`` (the wrapper's own
deadline expired and it killed the pytest process tree), ``interrupted`` (the
wrapper got SIGINT/SIGTERM and stopped the tree), ``spawn_failed``.

Exit status: pytest's own exit status whenever pytest exits by itself (a POSIX
signal death maps to 128+signal), 124 when the deadline killed it, 130 when the
wrapper was interrupted, 127 when pytest could not be started, 2 for a wrapper
usage error. The wrapper never retries anything.

Deadline: ``--timeout`` is relative to the wrapper's start; ``--deadline-epoch``
is an absolute Unix time, which lets CI reserve the end of a job's
``timeout-minutes`` for uploading the metrics (see .github/workflows/ci.yml). On
expiry the tree is stopped: POSIX sends SIGINT to pytest's process group (pytest
then still writes its JUnit file and session records), waits ``--grace`` seconds
and SIGKILLs the group; Windows terminates the tree at once (pytest shares the
wrapper's console group there, as it would without the wrapper, so no Ctrl+C can
be aimed at it alone). Records already flushed by the metrics plugin survive
either way.

Windows tree: pytest is put in a job object of its own, the stand-in for the
POSIX process group — membership outlives the root, so a child that ignored
Ctrl+C (``CREATE_NEW_PROCESS_GROUP``) is still reached after pytest itself exited
and was reaped, when ``taskkill /T`` can no longer walk to it. That is the normal
case on an interrupt: ``Popen.wait`` is not interruptible there, so the wrapper
only sees the Ctrl+C once pytest is gone. ``taskkill /T /F`` still runs after the
job kill for anything outside the job. If the job cannot be set up the run goes
on without it and ``runner.json`` records ``tree_job: false``.

Standalone by design: stdlib only, and it does not import bmad_loop. Child output
passes straight through to this process's stdout/stderr and is never recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUNNER_FILE = "runner.json"
MANIFEST_FILE = "manifest.json"
TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

EXIT_TIMED_OUT = 124
EXIT_SPAWN_FAILED = 127
EXIT_INTERRUPTED = 130
EXIT_USAGE = 2

# How long to wait for the direct child to be reaped after a tree kill.
_REAP_WAIT_S = 15.0
_TASKKILL_TIMEOUT_S = 30.0


class _Interrupted(Exception):
    """SIGINT/SIGTERM reached the wrapper while pytest was running."""


class _TreeJob:
    """Windows only: a job object holding pytest and everything it starts.

    Limits are just ``BREAKAWAY_OK``, so a child that explicitly asks to break
    away still can, as it could without the wrapper; there is no kill-on-close,
    so a run that exits normally leaves its processes exactly as it would
    unwrapped. Only a process pytest started before :meth:`open` assigned it —
    microseconds after the spawn, long before pytest starts anything — escapes."""

    def __init__(self, kernel32: Any, handle: int) -> None:
        self._kernel32 = kernel32
        self._handle: int | None = handle

    @classmethod
    def open(cls, pid: int) -> _TreeJob:
        """Create the job and assign ``pid`` to it; ``OSError`` on any failure."""
        import ctypes
        from ctypes import wintypes

        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        def fail(what: str) -> OSError:
            err = ctypes.get_last_error()
            return OSError(err, f"{what} failed (winerror {err})")

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise fail("CreateJobObjectW")
        job = cls(kernel32, handle)
        try:
            info = _Extended()
            info.BasicLimitInformation.LimitFlags = 0x800  # JOB_OBJECT_LIMIT_BREAKAWAY_OK
            if not kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):  # 9: JobObjectExtendedLimitInformation
                raise fail("SetInformationJobObject")
            # PROCESS_TERMINATE | PROCESS_SET_QUOTA. The pid cannot be reused
            # meanwhile: the caller's Popen still holds its process handle.
            process = kernel32.OpenProcess(0x0001 | 0x0100, False, pid)
            if not process:
                raise fail("OpenProcess")
            try:
                if not kernel32.AssignProcessToJobObject(handle, process):
                    raise fail("AssignProcessToJobObject")
            finally:
                kernel32.CloseHandle(process)
        except BaseException:
            job.close()
            raise
        return job

    def terminate(self) -> bool:
        """Kill every process still in the job. False if the call failed."""
        if self._handle is None:
            return False
        return bool(self._kernel32.TerminateJobObject(self._handle, 1))

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _write_runner(root: Path, record: dict[str, Any]) -> None:
    tmp = root / f".{RUNNER_FILE}.tmp"
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, root / RUNNER_FILE)


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest with --test-metrics-dir and record its process lifetime.",
        usage="%(prog)s --metrics-dir DIR [--timeout S | --deadline-epoch T] [--grace S] -- PYTEST_ARGS...",
    )
    parser.add_argument(
        "--metrics-dir", type=Path, required=True, help="fresh directory for all metrics files"
    )
    deadline = parser.add_mutually_exclusive_group()
    deadline.add_argument(
        "--timeout", type=float, help="seconds from now before the pytest tree is stopped"
    )
    deadline.add_argument(
        "--deadline-epoch", type=float, help="absolute Unix time at which the tree is stopped"
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=20.0,
        help="POSIX: seconds between SIGINT and SIGKILL on expiry (default 20; 0 kills at once)",
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="arguments for pytest, after --"
    )
    args = parser.parse_args(argv)
    if args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    if args.grace < 0 or (args.timeout is not None and args.timeout <= 0):
        parser.error("--grace must be >= 0 and --timeout > 0")
    return args


def _kill_tree(proc: subprocess.Popen[bytes], grace: float, job: _TreeJob | None = None) -> None:
    """Stop pytest and everything it started. Bounded: returns within
    ``grace + _REAP_WAIT_S`` (+ taskkill's own bound on Windows)."""
    if sys.platform == "win32":
        # The job first: it still holds the tree when the root is already
        # reaped, which taskkill's parent-pid walk cannot see past.
        if job is not None:
            job.terminate()
        try:
            subprocess.run(  # fixed argv, no shell
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_TASKKILL_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc.poll() is None:
            proc.kill()
    else:
        pgid = proc.pid  # start_new_session=True made pytest its group leader
        if grace > 0:
            try:
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        # SIGKILL the group even if pytest itself exited: workers it left behind
        # still hold the group id.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=_REAP_WAIT_S)
    except subprocess.TimeoutExpired:
        pass


def _raise_interrupted(signum: int, frame: object) -> None:
    raise _Interrupted(signum)


def _hold_signals_for_cleanup() -> None:
    """Stop a further SIGINT/SIGTERM from aborting the tree kill. A CI cancel
    sends SIGINT and then SIGTERM a few seconds later — inside the grace wait —
    and a second raise there would skip the group SIGKILL and leave the run
    record at ``running``. Cleanup is bounded, so nothing is lost by deferring:
    ``main``'s ``finally`` restores the caller's handlers."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)


def _summarize(root: Path) -> None:
    sys.path.insert(0, str(TESTS_DIR))
    try:
        import perf_report

        summary = perf_report.write_summary(root)
        sys.stdout.write("\n" + perf_report.render_text(summary))
        sys.stdout.flush()
    except Exception as exc:  # reporting must not mask pytest's exit status
        sys.stderr.write(
            f"run_test_benchmark: summary failed ({type(exc).__name__}: {exc}); "
            f"records in {root} are intact -- run `python tests/perf_report.py summarize {root}`\n"
        )
    finally:
        sys.path.remove(str(TESTS_DIR))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    root: Path = args.metrics_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / RUNNER_FILE).exists() or (root / MANIFEST_FILE).exists():
        sys.stderr.write(
            f"run_test_benchmark: {root} already holds a run; pass a fresh --metrics-dir\n"
        )
        return EXIT_USAGE

    start_wall = time.time()
    start_mono = time.monotonic()
    if args.deadline_epoch is not None:
        budget: float | None = args.deadline_epoch - start_wall
    else:
        budget = args.timeout
    record: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "state": "running",
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": sys.platform,
        "start_wall": start_wall,
        "start_mono": start_mono,
        "budget_seconds": budget,
        "grace_seconds": args.grace,
    }
    _write_runner(root, record)

    def finish(state: str, exit_code: int | None, **extra: Any) -> None:
        end_mono = time.monotonic()
        record.update(
            state=state,
            exit_code=exit_code,
            end_wall=time.time(),
            end_mono=end_mono,
            elapsed_seconds=round(end_mono - start_mono, 3),
            **extra,
        )
        _write_runner(root, record)

    if budget is not None and budget <= 0:
        sys.stderr.write(
            f"run_test_benchmark: deadline already passed ({budget:.0f}s); not starting pytest\n"
        )
        finish("timed_out", None, child_started=False)
        _summarize(root)
        return EXIT_TIMED_OUT

    command = [sys.executable, "-m", "pytest", f"--test-metrics-dir={root}", *args.pytest_args]
    # POSIX: a session of its own, so the whole tree can be signalled as a group.
    # Windows: deliberately NOT a new process group — that would disable Ctrl+C
    # for pytest and everything it starts, changing the environment under test
    # (an accidental CTRL_C_EVENT would stop aborting the run). `taskkill /T`
    # needs no group.
    popen_kwargs: dict[str, Any] = {} if sys.platform == "win32" else {"start_new_session": True}

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, _raise_interrupted)
    job: _TreeJob | None = None
    try:
        try:
            proc = subprocess.Popen(command, **popen_kwargs)  # fixed argv, no shell
        except OSError as exc:
            sys.stderr.write(
                f"run_test_benchmark: cannot start pytest: {type(exc).__name__}: {exc}\n"
            )
            finish("spawn_failed", None, error=type(exc).__name__)
            _summarize(root)
            return EXIT_SPAWN_FAILED
        record["child_pid"] = proc.pid
        if sys.platform == "win32":
            try:
                job = _TreeJob.open(proc.pid)
            except OSError as exc:
                sys.stderr.write(
                    f"run_test_benchmark: no job object for the pytest tree ({exc}); "
                    "a stop reaches only what taskkill /T can still find\n"
                )
            record["tree_job"] = job is not None
        _write_runner(root, record)
        try:
            returncode = proc.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            _hold_signals_for_cleanup()
            sys.stderr.write(
                f"run_test_benchmark: deadline reached after {budget:.0f}s; stopping pytest\n"
            )
            _kill_tree(proc, args.grace, job)
            finish("timed_out", proc.returncode, child_reaped=proc.returncode is not None)
            _summarize(root)
            return EXIT_TIMED_OUT
        except _Interrupted:
            _hold_signals_for_cleanup()
            if sys.platform == "win32":
                # A console Ctrl+C already reached pytest too (same console
                # group): give it the grace to finish before the tree kill.
                try:
                    proc.wait(timeout=args.grace)
                except subprocess.TimeoutExpired:
                    pass
            _kill_tree(proc, args.grace, job)
            finish("interrupted", proc.returncode, child_reaped=proc.returncode is not None)
            _summarize(root)
            return EXIT_INTERRUPTED
    finally:
        if job is not None:
            job.close()  # no kill-on-close: this frees the handle, nothing more
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    finish("exited", returncode)
    _summarize(root)
    if returncode < 0:
        return 128 - returncode
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
