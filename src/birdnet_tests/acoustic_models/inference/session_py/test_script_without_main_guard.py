"""A script that calls birdnet outside of ``if __name__ == "__main__":``.

Every pipeline process imports the main script again, so such a script runs
birdnet in each of them, where no process can be started. It cannot work, but it
has to fail at once and say why: the error used to blame the memory. Run here
as real scripts, because only a main module that is a file is imported again.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

import birdnet_tests
from birdnet.globals import ENV_VAR_START_METHOD
from birdnet_tests.fake_acoustic_backend import fake_predict_session

pytestmark = pytest.mark.no_tf

_MAIN_GUARD = 'if __name__ == "__main__"'
# A healthy run of these scripts takes a few seconds; the bound only has to
# tell a failure that reports itself from one that hangs.
_TIMEOUT_S = 180

_UNGUARDED = """\
import sys
from pathlib import Path

import numpy as np

from birdnet_tests.fake_acoustic_backend import fake_predict_session

with fake_predict_session(Path(sys.argv[1]), n_workers=1) as session:
  session.run_arrays((np.ones(3 * 48_000, dtype=np.float32), 48_000))
print("finished")
"""

_GUARDED = """\
import sys
from pathlib import Path

import numpy as np

from birdnet_tests.fake_acoustic_backend import fake_predict_session

if __name__ == "__main__":
  with fake_predict_session(Path(sys.argv[1]), n_workers=1) as session:
    session.run_arrays((np.ones(3 * 48_000, dtype=np.float32), 48_000))
  print("finished")
"""

# Only what starts processes needs the guard: every process builds this session
# again while it imports the script, and none may trip over that.
_BUILT_AT_MODULE_LEVEL = """\
import sys
from pathlib import Path

import numpy as np

from birdnet_tests.fake_acoustic_backend import fake_predict_session

session = fake_predict_session(Path(sys.argv[1]), n_workers=1)

if __name__ == "__main__":
  with session:
    session.run_arrays((np.ones(3 * 48_000, dtype=np.float32), 48_000))
  print("finished")
"""


def _run(tmp_path: Path, source: str) -> subprocess.CompletedProcess[str]:
  script = tmp_path / "script.py"
  script.write_text(source, encoding="utf-8")
  env = dict(os.environ)
  # The test package is importable here through pytest, not in a fresh interpreter.
  env["PYTHONPATH"] = os.pathsep.join(
    p
    for p in (str(Path(birdnet_tests.__file__).parents[1]), env.get("PYTHONPATH"))
    if p
  )
  env[ENV_VAR_START_METHOD] = "spawn"
  return subprocess.run(
    [sys.executable, str(script), str(tmp_path)],
    capture_output=True,
    text=True,
    env=env,
    timeout=_TIMEOUT_S,
    check=False,
  )


def test_an_unguarded_script_fails_naming_the_main_guard(tmp_path: Path) -> None:
  result = _run(tmp_path, _UNGUARDED)

  assert result.returncode != 0
  assert "finished" not in result.stdout
  # The last traceback is the script's own, the error its author is left with;
  # the pipeline processes print theirs before they exit.
  final_error = result.stderr.rsplit("Traceback (most recent call last)", 1)[-1]
  assert _MAIN_GUARD in final_error


@pytest.mark.parametrize(
  "source", [_GUARDED, _BUILT_AT_MODULE_LEVEL], ids=["guarded", "built at module level"]
)
def test_a_script_that_guards_what_starts_processes_runs(
  tmp_path: Path, source: str
) -> None:
  result = _run(tmp_path, source)

  assert result.returncode == 0, result.stderr
  assert "finished" in result.stdout


def test_a_session_starts_nothing_in_a_process_importing_the_main_script(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  before = {child.pid for child in psutil.Process().children()}
  # The flag multiprocessing sets on a new process around that import.
  monkeypatch.setattr(mp.current_process(), "_inheriting", True, raising=False)
  session = fake_predict_session(tmp_path, n_workers=1)

  with pytest.raises(RuntimeError, match=_MAIN_GUARD):
    session.__enter__()

  assert session._resource_manager is None
  assert {child.pid for child in psutil.Process().children()} - before == set()
