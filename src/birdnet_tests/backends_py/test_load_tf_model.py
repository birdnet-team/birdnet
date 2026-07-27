import platform

import pytest

from birdnet.acoustic.models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.core.backends import load_tf_model
from birdnet.utils.local_data import get_model_path
from birdnet_tests.helper import ensure_litert_or_skip, use_fork_or_skip


def test_load_tf_and_litert_after_each_other_is_not_possible() -> None:
  pytest.skip("Does not throw an exception with ai-edge-litert==2.1.4. @stefantaubert")
  # not marked as @pytest.mark.litert because TF is loaded first
  # other tests with litert marking would fail then
  ensure_litert_or_skip()
  # needs fork because it the backend is loaded in the main process
  use_fork_or_skip()

  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  model_path = get_model_path("acoustic", "2.4", "tf", "fp32")

  # Load TF model
  model_tf = load_tf_model(model_path, library="tflite", allocate_tensors=False)
  assert model_tf is not None

  # Loading Litert model after TF fails
  # Error message differs between platforms
  # only on Python 3.13
  is_python_313 = ("3", "13") <= platform.python_version_tuple() < ("3", "14")

  if is_python_313:
    with pytest.raises(ImportError):
      load_tf_model(model_path, library="litert", allocate_tensors=False)
  else:
    pytest.skip("Loading Litert model after TF fails only on Python 3.13")


@pytest.mark.litert
def test_load_litert_and_tf_after_each_other_is_possible() -> None:
  ensure_litert_or_skip()
  # needs fork because it the backend is loaded in the main process
  use_fork_or_skip()

  AcousticTFDownloaderV2_4.get_model_path_and_labels("en_us", "fp32")
  model_path = get_model_path("acoustic", "2.4", "tf", "fp32")

  model_litert = load_tf_model(model_path, library="litert", allocate_tensors=False)
  assert model_litert is not None

  # Load TF model
  # On macOS the mixed load is blocked only on Intel; Apple Silicon allows it with
  # current ai-edge-litert (cf. the skipped test_load_tf_and_litert_..._not_possible).
  if platform.system() == "Darwin" and platform.machine() == "x86_64":
    # Loading TF model after liteRT fails on Intel macOS
    with pytest.raises(ImportError):
      load_tf_model(model_path, library="tflite", allocate_tensors=False)
  else:
    model_tf = load_tf_model(model_path, library="tflite", allocate_tensors=False)
    assert model_tf is not None
