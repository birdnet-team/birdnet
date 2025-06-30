import logging
import os
from pathlib import Path
from typing import Any, Callable, final

import absl.logging
import numpy as np

from birdnet_v2.acoustic_models.base import AcousticInferenceBackend


class AcousticPBBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._logical_device: Any | None = None
    self._infer_fn: Callable | None = None

  @final
  def lazy_load(self, device_name: str) -> None:
    assert "GPU" in device_name or "CPU" in device_name

    absl_verbosity_before = absl.logging.get_verbosity()
    absl.logging.set_verbosity(absl.logging.ERROR)
    tf_verbosity_before = logging.getLogger("tensorflow").level
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.random.set_seed(0)

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

    audio_model = tf.saved_model.load(self._model_path)

    absl.logging.set_verbosity(absl_verbosity_before)
    logging.getLogger("tensorflow").setLevel(tf_verbosity_before)

    # _SignatureMap({'basic': <ConcreteFunction (*, inputs: TensorSpec(shape=(None, 144000), dtype=tf.float32, name='inputs')) -> Dict[['scores', TensorSpec(shape=(None, 6522), dtype=tf.float32, name='scores')]] at 0x7BD844349190>, 'embeddings': <ConcreteFunction (*, inputs: TensorSpec(shape=(None, 144000), dtype=tf.float32, name='inputs')) -> Dict[['embeddings', TensorSpec(shape=(None, 1024), dtype=tf.float32, name='embeddings')]] at 0x7BD8684EBC50>})
    self._infer_fn = audio_model.signatures["basic"]  # type: ignore

  @final
  def infer(self, batch: np.ndarray) -> np.ndarray:
    assert self._logical_device is not None
    assert self._infer_fn is not None
    from tensorflow import Tensor, device, float32

    with device(self._logical_device.name):  # type: ignore
      # prediction = self._audio_model.basic(batch)["scores"]
      predictions = self._infer_fn(inputs=batch)
    scores: Tensor = predictions["scores"]
    assert scores.dtype == float32
    scores_np = scores.numpy()  # type: ignore
    assert scores_np.dtype == np.float32
    return scores_np
