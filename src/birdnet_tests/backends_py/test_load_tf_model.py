import platform

import pytest

from birdnet.acoustic_models.v2_4.tf import AcousticTFDownloaderV2_4
from birdnet.backends import load_tf_model
from birdnet.local_data import get_model_path
from birdnet_tests.helper import ensure_litert_or_skip, use_fork_or_skip


def test_load_tf_and_litert_after_each_other_is_not_possible() -> None:
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
  with pytest.raises(ImportError):
    load_tf_model(model_path, library="litert", allocate_tensors=False)


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
  # on macOS it can't be loaded after liteRT
  if platform.system() == "Darwin":
    # Loading Litert model after TF fails
    with pytest.raises(ImportError):
      load_tf_model(model_path, library="tflite", allocate_tensors=False)
  else:
    model_tf = load_tf_model(model_path, library="tflite", allocate_tensors=False)
    assert model_tf is not None
