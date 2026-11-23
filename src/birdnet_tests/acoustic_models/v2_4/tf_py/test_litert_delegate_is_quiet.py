"""Loading a model on LiteRT's default delegate writes nothing to stderr.

The interpreter announces the delegate from native code as it allocates
(``INFO: Created TensorFlow Lite XNNPACK delegate for CPU.``), once in every
worker. A fresh interpreter per run: the line is written once per process.
"""

import os
import subprocess
import sys

import pytest

import birdnet
from birdnet.globals import ENV_VAR_TF_VERBOSE
from birdnet_tests.helper import ensure_litert_or_skip

pytestmark = pytest.mark.litert

_SNIPPET = (
  "import sys\n"
  "from pathlib import Path\n"
  "from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4\n"
  "AcousticTFBackendFP32V2_4(\n"
  "  Path(sys.argv[1]), 'CPU', False, inference_library='litert'\n"
  ").load()\n"
)
_LOAD_TIMEOUT_S = 300


def _stderr_of_a_load(model_path: str, verbose: bool) -> str:
  env = dict(os.environ)
  if verbose:
    env[ENV_VAR_TF_VERBOSE] = "1"
  else:
    env.pop(ENV_VAR_TF_VERBOSE, None)
  done = subprocess.run(
    [sys.executable, "-c", _SNIPPET, model_path],
    capture_output=True,
    text=True,
    env=env,
    timeout=_LOAD_TIMEOUT_S,
    check=False,
  )
  assert done.returncode == 0, done.stderr
  return done.stderr


def test_loading_on_the_delegate_writes_nothing_to_stderr() -> None:
  ensure_litert_or_skip()
  model = birdnet.load("acoustic", "2.4", "tf", library="litert")
  model_path = str(model.model_path)

  if "delegate" not in _stderr_of_a_load(model_path, verbose=True):
    pytest.skip("This LiteRT does not announce its delegate; nothing to suppress")

  assert _stderr_of_a_load(model_path, verbose=False) == ""
