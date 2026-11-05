import os
import subprocess
import sys

import pytest

from birdnet.globals import ENV_VAR_TF_VERBOSE

# A fresh interpreter per run: the banner is emitted once per process, when
# TensorFlow is first imported, so it cannot be observed in this one.
_SNIPPET = "from birdnet.core.backends import import_tf; import_tf()"

_IMPORT_TIMEOUT_S = 300


def _start(verbose: bool) -> subprocess.Popen[str]:
  env = dict(os.environ)
  if verbose:
    env[ENV_VAR_TF_VERBOSE] = "1"
  else:
    env.pop(ENV_VAR_TF_VERBOSE, None)
  return subprocess.Popen(
    [sys.executable, "-c", _SNIPPET],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    env=env,
  )


def _stderr(process: subprocess.Popen[str]) -> str:
  _, stderr = process.communicate(timeout=_IMPORT_TIMEOUT_S)
  return stderr


def test_importing_tensorflow_writes_nothing_to_stderr() -> None:
  """The point of the whole feature: a plain import prints nothing.

  Guarded against passing for the wrong reason — if TensorFlow is silent here
  even with suppression off, there is nothing to suppress and nothing to prove.
  """
  # Side by side: two TensorFlow imports at once are the lane's everyday load.
  loud = _start(verbose=True)
  quiet = _start(verbose=False)
  try:
    loud_stderr = _stderr(loud)
    quiet_stderr = _stderr(quiet)
  finally:
    for process in (loud, quiet):
      if process.poll() is None:
        process.kill()
        process.wait()
  assert loud.returncode == 0, loud_stderr
  if not loud_stderr.strip():
    pytest.skip("TensorFlow printed nothing on import; nothing to suppress here")

  assert quiet.returncode == 0, quiet_stderr
  assert quiet_stderr == "", f"expected no output, got:\n{quiet_stderr}"
