"""Unit tests for `scripts/sync_version.py`'s `check()` — the `version-sync` CI gate.

The gate's job is to refuse a release whose version fields disagree. What these cover
is the failure mode that is *invisible* when it happens: `check()` reading a
marketplace.json whose `plugins` list it cannot find, iterating zero times, and
printing "ok: every version field agrees" — announcing agreement it never verified.
A gate that is green for both "correct" and "unreadable" carries no information.

Every negative case here is paired with a positive control (a genuinely stale version
must still redden), because a `check() != 0` assertion passes for every reason a
non-zero could arise — including a bug in the fixture.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import sync_version  # noqa: E402


@pytest.fixture
def marketplace(tmp_path, monkeypatch):
    """Point `check()` at a throwaway marketplace.json, leaving every other file real.

    Returns a writer taking the parsed-JSON object, so a test names the corruption it
    is injecting rather than a blob of literal JSON.
    """
    target = tmp_path / "marketplace.json"
    monkeypatch.setattr(sync_version, "MARKETPLACE", target)
    canonical = sync_version.read_canonical()

    def write(obj) -> None:
        target.write_text(json.dumps(obj), encoding="utf-8")

    write({"plugins": [{"name": "bmad-loop", "version": canonical}]})
    return write


def test_agreeing_versions_pass(marketplace):
    """Baseline: the shape the repo actually ships is green."""
    assert sync_version.check() == 0


def test_stale_plugin_version_is_caught(marketplace):
    """Positive control. Without this, every assertion below could be passing
    because the gate is broken outright rather than because the guard works."""
    marketplace({"plugins": [{"name": "bmad-loop", "version": "0.0.1-stale"}]})
    assert sync_version.check() == 1


@pytest.mark.parametrize(
    ("label", "doc"),
    [
        ("renamed key", {"plugin": [{"name": "bmad-loop", "version": "0.0.1-stale"}]}),
        ("missing key", {"name": "bmad marketplace"}),
        ("empty list", {"plugins": []}),
        ("wrong type", {"plugins": {}}),
        ("null", {"plugins": None}),
    ],
)
def test_unusable_plugins_list_fails_instead_of_passing_vacuously(marketplace, label, doc):
    """The defect this file exists for.

    Measured against the pre-guard code, these split into two failure modes, and the
    guard's job is to turn both into one clean drift report:

    * ``renamed key`` / ``missing key`` / ``empty list`` / ``wrong type`` returned **0**
      while printing "ok: every version field agrees on <v>" — a silent vacuous pass.
    * ``null`` raised ``TypeError`` from ``enumerate(None)`` — fail-closed, but as a
      crashed job rather than the drift report this gate exists to emit.

    ``renamed key`` is deliberately loaded with a *stale* version too, so a green result
    would mean the gate waved through real drift, not merely an empty plugin list.
    """
    marketplace(doc)
    assert sync_version.check() == 1, f"{label}: check() passed with no plugins to inspect"


def test_non_object_plugin_entry_is_reported_not_raised(marketplace):
    """A string where an object belongs used to hit `.get` on a str and raise
    AttributeError. Exiting non-zero via traceback is technically fail-closed, but it
    reads as a crashed job rather than the drift report this gate is supposed to emit."""
    marketplace({"plugins": ["bmad-loop"]})
    assert sync_version.check() == 1


def test_second_plugin_stale_is_caught(marketplace):
    """The loop must not stop at the first agreeing entry."""
    canonical = sync_version.read_canonical()
    marketplace(
        {
            "plugins": [
                {"name": "bmad-loop", "version": canonical},
                {"name": "other", "version": "0.0.1-stale"},
            ]
        }
    )
    assert sync_version.check() == 1


# --- parse_canonical: the one owner of the __version__ parse ---------------- #
# release.py reads `git show <commit>:__init__.py` blobs through it, so it must
# agree with read_canonical on the checkout and answer None, not exit, on a miss.
def test_parse_canonical_agrees_with_read_canonical():
    assert sync_version.parse_canonical(sync_version.INIT.read_text()) == (
        sync_version.read_canonical()
    )


def test_parse_canonical_is_none_without_a_version():
    assert sync_version.parse_canonical('"""no version here."""\n') is None


# --- stamp() writes LF on every platform (DW-517) --------------------------- #
def test_stamp_writes_lf_under_windows_newline_translation(
    tmp_path, monkeypatch, emulate_windows_newlines
):
    """A Windows release cut must not rewrite the stamped LF files as CRLF —
    `.gitattributes` has no `text=auto` rule to normalize them back."""
    stamped = {}
    for name in ("INIT", "PYPROJECT", "CANONICAL_MODULE_YAML", "MARKETPLACE"):
        src = getattr(sync_version, name)
        dst = tmp_path / f"{name.lower()}{src.suffix}"
        shutil.copyfile(src, dst)
        monkeypatch.setattr(sync_version, name, dst)
        stamped[name] = dst
    monkeypatch.setattr(sync_version, "ROOT_MODULE_YAML", tmp_path / "root-module.yaml")
    monkeypatch.setattr(sync_version, "_relock", lambda: None)
    emulate_windows_newlines()

    sync_version.stamp("99.0.0")

    for name, path in stamped.items():
        data = path.read_bytes()
        # Control: the file was actually rewritten, so "no CRLF" is not vacuous.
        assert b"99.0.0" in data, name
        assert b"\r\n" not in data, name
