import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, final

import numpy as np

from birdnet.acoustic_models.base import AcousticInferenceBackend
from birdnet.io_lock import IOLockHandler
from birdnet.logging_utils import get_logger


class AcousticPBBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._cached_logical_device: Any | None = None
    self._infer_fn: Callable | None = None
    self._cached_device_name: str | None = None

  @final
  def load(self, io_lock_handler: IOLockHandler) -> None:
    import absl.logging

    absl_verbosity_before = absl.logging.get_verbosity()
    absl.logging.set_verbosity(absl.logging.ERROR)
    tf_verbosity_before = logging.getLogger("tensorflow").level
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.random.set_seed(0)

    with io_lock_handler:
      start = time.perf_counter()
      audio_model = tf.saved_model.load(self._model_path)
      end = time.perf_counter()
    logger = get_logger(__name__)
    logger.debug(f"Model loaded from {self._model_path} in {end - start:.2f} seconds.")

    absl.logging.set_verbosity(absl_verbosity_before)
    logging.getLogger("tensorflow").setLevel(tf_verbosity_before)

    # _SignatureMap({'basic': <ConcreteFunction (*, inputs: TensorSpec(shape=(None, 144000), dtype=tf.float32, name='inputs')) -> Dict[['scores', TensorSpec(shape=(None, 6522), dtype=tf.float32, name='scores')]] at 0x7BD844349190>, 'embeddings': <ConcreteFunction (*, inputs: TensorSpec(shape=(None, 144000), dtype=tf.float32, name='inputs')) -> Dict[['embeddings', TensorSpec(shape=(None, 1024), dtype=tf.float32, name='embeddings')]] at 0x7BD8684EBC50>})
    self._infer_fn = audio_model.signatures["basic"]  # type: ignore

  def _set_logical_device(self, device_name: str) -> None:
    assert "GPU" in device_name or "CPU" in device_name
    import tensorflow as tf

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
      if tf.config.experimental.get_memory_growth(physical_gpu_device) is False:
        tf.config.experimental.set_memory_growth(physical_gpu_device, True)
      self._cached_logical_device = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ][0]

    elif "CPU" in device_name:
      all_devices_with_name: list = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ]
      if len(all_devices_with_name) == 0:
        raise ValueError(f"No CPU with name '{device_name}' found!")
      self._cached_logical_device = all_devices_with_name[0]
    else:
      raise ValueError(f"Unsupported device name: {device_name}")

  @final
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray:
    if self._cached_device_name is None or self._cached_device_name != device_name:
      self._set_logical_device(device_name)
      self._cached_device_name = device_name

    assert self._cached_logical_device is not None
    assert self._infer_fn is not None
    from tensorflow import Tensor, device, float32

    with device(self._cached_logical_device.name):  # type: ignore
      # prediction = self._audio_model.basic(batch)["scores"]
      predictions = self._infer_fn(inputs=batch)
    scores: Tensor = predictions["scores"]
    assert scores.dtype == float32
    scores_np = scores.numpy()  # type: ignore
    assert scores_np.dtype == np.float32
    return scores_np
