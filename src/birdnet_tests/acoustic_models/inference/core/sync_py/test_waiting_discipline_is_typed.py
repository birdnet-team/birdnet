"""The next raw queue read or child-side event wait is a type error.

Ruff's banned-API check cannot see ``q.get(..)`` on a local, so the rule is a
type: a child-to-parent queue is typed without ``get`` and a child's event
without ``wait``, and mypy rejects the call. CI does not run mypy on its own,
so this test does: one cold run over the package, filtered to the two protocol
errors, and over snippets that pin the protocols themselves. A cold run
re-analyzes the whole import closure, minutes of one core on the macOS Intel
runners and enough to fail the timing tests beside it; type checking does not
depend on the platform, so only the Linux lanes run it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import birdnet

pytestmark = [
  pytest.mark.no_tf,
  pytest.mark.skipif(sys.platform != "linux", reason="the Linux lanes run mypy"),
]

pytest.importorskip("mypy")

_SRC = Path(birdnet.__file__).resolve().parents[1]
_CONFIG = _SRC.parent / "pyproject.toml"
_PROTOCOL_ERRORS = (
  '"ChildToParentQueue" has no attribute',
  '"ChildSideEvent" has no attribute',
)

_SNIPPETS = {
  "raw_read": """
from birdnet.acoustic.inference.core.sync import ChildToParentQueue

def read(q: ChildToParentQueue) -> object:
  return q.get(timeout=1.0)
""",
  "child_wait": """
from birdnet.acoustic.inference.core.sync import ChildSideEvent

def wait(e: ChildSideEvent) -> bool:
  return e.wait(timeout=1.0)
""",
  "allowed": """
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
""",
}


def _mypy(cache_root: Path, *targets: str) -> str:
  """mypy on the birdnet this test imported; one error per line."""
  command = [
    sys.executable,
    "-m",
    "mypy",
    "--follow-imports=silent",
    "--no-error-summary",
    # One line per error, whatever the project config says about pretty output.
    "--no-pretty",
    "--cache-dir",
    str(cache_root / ".mypy_cache"),
  ]
  if _CONFIG.exists():
    command += ["--config-file", str(_CONFIG)]
  result = subprocess.run(
    [*command, *targets],
    capture_output=True,
    text=True,
    check=False,
    env={**os.environ, "MYPYPATH": str(_SRC)},
  )
  # 2 is mypy not running at all, which the filters below must not hide.
  assert result.returncode in (0, 1), result.stdout + result.stderr
  return result.stdout


@pytest.fixture(scope="module")
def errors_by_file(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
  """The package's and the snippets' error lines, keyed by file.

  mypy takes a package or files, not both, so two runs on one cache: the
  package pays for the import closure once and the snippets reuse it.
  """
  root = tmp_path_factory.mktemp("typed")
  snippets = {name: root / f"{name}.py" for name in _SNIPPETS}
  for name, path in snippets.items():
    path.write_text(_SNIPPETS[name])
  out = _mypy(root, "-p", "birdnet.acoustic.inference")
  out += _mypy(root, *map(str, snippets.values()))
  grouped: dict[str, list[str]] = {name: [] for name in [*_SNIPPETS, "package"]}
  for line in out.splitlines():
    for name, path in snippets.items():
      if line.startswith(str(path)):
        grouped[name].append(line)
        break
    else:
      grouped["package"].append(line)
  return grouped


def test_the_pipeline_package_has_no_raw_read_or_child_side_wait(
  errors_by_file: dict[str, list[str]],
) -> None:
  # The package has known mypy errors elsewhere; only the two protocol errors
  # are this test's business.
  offending = [
    line
    for line in errors_by_file["package"]
    if any(error in line for error in _PROTOCOL_ERRORS)
  ]
  assert not offending, "\n".join(offending)


def test_a_raw_queue_read_is_a_type_error(
  errors_by_file: dict[str, list[str]],
) -> None:
  wanted = '"ChildToParentQueue" has no attribute "get"'
  assert any(wanted in line for line in errors_by_file["raw_read"]), errors_by_file


def test_a_child_side_event_wait_is_a_type_error(
  errors_by_file: dict[str, list[str]],
) -> None:
  wanted = '"ChildSideEvent" has no attribute "wait"'
  assert any(wanted in line for line in errors_by_file["child_wait"]), errors_by_file


def test_the_sanctioned_operations_type_check(
  errors_by_file: dict[str, list[str]],
) -> None:
  assert errors_by_file["allowed"] == []
