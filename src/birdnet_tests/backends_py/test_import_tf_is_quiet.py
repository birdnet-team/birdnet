import os
import subprocess
import sys

import pytest

from birdnet.globals import ENV_VAR_TF_VERBOSE

# A fresh interpreter per run: the banner is emitted once per process, when
# TensorFlow is first imported, so it cannot be observed in this one.
_SNIPPET = "from birdnet.core.backends import import_tf; import_tf()"

_IMPORT_TIMEOUT_S = 300


def _run(verbose: bool) -> subprocess.CompletedProcess[str]:
  env = dict(os.environ)
  if verbose:
    env[ENV_VAR_TF_VERBOSE] = "1"
  else:
    env.pop(ENV_VAR_TF_VERBOSE, None)
  return subprocess.run(
    [sys.executable, "-c", _SNIPPET],
    capture_output=True,
    text=True,
    env=env,
    timeout=_IMPORT_TIMEOUT_S,
    check=False,
  )


def test_importing_tensorflow_writes_nothing_to_stderr() -> None:
  """The point of the whole feature: a plain import prints nothing.

  Guarded against passing for the wrong reason — if TensorFlow is silent here
  even with suppression off, there is nothing to suppress and nothing to prove.
  """
  loud = _run(verbose=True)
  assert loud.returncode == 0, loud.stderr
  if not loud.stderr.strip():
    pytest.skip("TensorFlow printed nothing on import; nothing to suppress here")

  quiet = _run(verbose=False)
  assert quiet.returncode == 0, quiet.stderr
  assert quiet.stderr == "", f"expected no output, got:\n{quiet.stderr}"
