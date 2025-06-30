import sys
from pathlib import Path
from typing import Any, final

import numpy as np

from birdnet_v2.acoustic_models.base import AcousticInferenceBackend


class AcousticPBBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path, device: str) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._audio_model = None
    self._device_name = device
    self._logical_device: Any | None = None

  @final
  def lazy_load(self, device_name: str) -> None:
    assert "GPU" in device_name or "CPU" in device_name
    import tensorflow as tf

    tf.random.set_seed(0)
    tf.get_logger().setLevel("ERROR")
    tf.debugging.set_log_device_placement(False)

    device: tf.config.LogicalDevice | None = None
    if "GPU" in device_name:
      physical_devices = tf.config.list_physical_devices("GPU")
      if len(physical_devices) == 0:
        raise ValueError(
          "No GPU found! Please check your TensorFlow installation and ensure that a GPU is available."
        )

      gpus_with_name = [gpu for gpu in physical_devices if device_name in gpu.name]

      if len(gpus_with_name) == 0:
        raise ValueError(f"No GPU with name '{device_name}' found!")

      physical_gpu_device = gpus_with_name[0]
      tf.config.experimental.set_memory_growth(physical_gpu_device, True)
      device = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ][0]

    if "CPU" in device_name:
      all_devices_with_name: list = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ]
      if len(all_devices_with_name) == 0:
        raise ValueError(f"No CPU with name '{device_name}' found!")
      device = all_devices_with_name[0]

    assert device is not None
    self._logical_device = device

    assert self._audio_model is None
    self._audio_model = tf.saved_model.load(self._model_path)

  @final
  def infer(self, batch: np.ndarray) -> np.ndarray:
    assert self._audio_model is not None
    assert self._logical_device is not None
    # basic_fn = self._audio_model.signatures["basic"]  # oder "basic"

    # keine Retrace-Warnungen, weil wir eine Concrete-Function benutzen
    # prediction = basic_fn(inputs=batch)
    # prediction = prediction["scores"]
    import tensorflow as tf

    with tf.device(self._logical_device.name):  # type: ignore
      prediction = self._audio_model.basic(batch)["scores"]
    prediction_np = prediction.numpy()
    assert prediction_np.dtype == np.float32
    return prediction_np
