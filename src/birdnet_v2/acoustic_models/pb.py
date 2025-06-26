from pathlib import Path
from typing import final

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from birdnet_v2.acoustic_models.base import AcousticInferenceBackend


class AcousticPBBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path, device: str) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._audio_model = None
    self._device_name = device
    self._device: tf.config.LogicalDevice | None = None

  @final
  def lazy_load(self) -> None:
    assert self._audio_model is None

    all_devices_with_name = tf.config.list_logical_devices(self._device_name)
    assert len(all_devices_with_name) == 1
    device: tf.config.LogicalDevice = all_devices_with_name[0]
    self._device = device

    self._audio_model = tf.saved_model.load(self._model_path)

  @final
  def infer(self, batch: np.ndarray) -> np.ndarray:
    assert self._audio_model is not None
    assert self._device is not None

    with tf.device(self._device):  # type: ignore
      prediction: Tensor = self._audio_model.basic(batch)["scores"]
    prediction_np = prediction.numpy()
    assert prediction_np.dtype == np.float32
    return prediction_np
