"""The next raw queue read or child-side event wait is a type error.

Ruff's banned-API check cannot see ``q.get(..)`` on a local, so the rule is a
type: a child-to-parent queue is typed without ``get`` and a child's event
without ``wait``, and mypy rejects the call. CI does not run mypy on its own,
so this test does: on the package, filtered to the two protocol errors, and on
snippets that pin the protocols themselves.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import birdnet

pytestmark = pytest.mark.no_tf

pytest.importorskip("mypy")

_SRC = Path(birdnet.__file__).resolve().parents[1]
_CONFIG = _SRC.parent / "pyproject.toml"
_PROTOCOL_ERRORS = (
  '"ChildToParentQueue" has no attribute',
  '"ChildSideEvent" has no attribute',
)

_RAW_READ = """
from birdnet.acoustic.inference.core.sync import ChildToParentQueue

def read(q: ChildToParentQueue) -> object:
  return q.get(timeout=1.0)
"""

_CHILD_WAIT = """
from birdnet.acoustic.inference.core.sync import ChildSideEvent

def wait(e: ChildSideEvent) -> bool:
  return e.wait(timeout=1.0)
"""

_ALLOWED = """
from birdnet.acoustic.inference.core.sync import (
  ChildSideEvent,
  ChildToParentQueue,
  ThreadedQueueReader,
  readable,
)

def use(q: ChildToParentQueue, e: ChildSideEvent) -> None:
  q.put(1)
  q.put_nowait(2)
  q.cancel_join_thread()
  if not e.is_set():
    e.set()
  reader: ThreadedQueueReader = ThreadedQueueReader(q, "pump")
  reader.stop()
  readable(q).get_nowait()
"""


def _mypy(tmp_path: Path, *targets: str) -> tuple[int, str]:
  """mypy on the birdnet that this test imported, with a cold cache."""
  command = [
    sys.executable,
    "-m",
    "mypy",
    "--follow-imports=silent",
    "--no-error-summary",
    # One line per error, whatever the project config says about pretty output.
    "--no-pretty",
    "--cache-dir",
    str(tmp_path / ".mypy_cache"),
  ]
  if _CONFIG.exists():
    command += ["--config-file", str(_CONFIG)]
  env = {**os.environ, "MYPYPATH": str(_SRC)}
  result = subprocess.run(
    [*command, *targets], capture_output=True, text=True, check=False, env=env
  )
  return result.returncode, result.stdout


def _snippet(tmp_path: Path, name: str, source: str) -> tuple[int, str]:
  snippet = tmp_path / f"{name}.py"
  snippet.write_text(source)
  return _mypy(tmp_path, str(snippet))


def test_the_pipeline_package_has_no_raw_read_or_child_side_wait(
  tmp_path: Path,
) -> None:
  # The package has known mypy errors elsewhere; only the two protocol errors
  # are this test's business.
  _, out = _mypy(tmp_path, "-p", "birdnet.acoustic.inference")
  offending = [
    line for line in out.splitlines() if any(e in line for e in _PROTOCOL_ERRORS)
  ]
  assert not offending, "\n".join(offending)
  assert "birdnet" in out or out == "", out


def test_a_raw_queue_read_is_a_type_error(tmp_path: Path) -> None:
  code, out = _snippet(tmp_path, "raw_read", _RAW_READ)
  assert code != 0
  assert '"ChildToParentQueue" has no attribute "get"' in out, out


def test_a_child_side_event_wait_is_a_type_error(tmp_path: Path) -> None:
  code, out = _snippet(tmp_path, "child_wait", _CHILD_WAIT)
  assert code != 0
  assert '"ChildSideEvent" has no attribute "wait"' in out, out


def test_the_sanctioned_operations_type_check(tmp_path: Path) -> None:
  code, out = _snippet(tmp_path, "allowed", _ALLOWED)
  assert code == 0, out
