from pathlib import Path
from typing import final

import numpy as np


from birdnet_v2.acoustic_models.base import AcousticInferenceBackend


class AcousticPBBackend(AcousticInferenceBackend):
  def __init__(self, model_path: Path, device: str) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._audio_model = None
    self._device_name = device
    self._device: None = None

  @final
  def lazy_load(self) -> None:
    import tensorflow as tf
    
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
      tf.config.experimental.set_memory_growth(gpu_instance, True)
        
    tf.random.set_seed(0)
    tf.get_logger().setLevel("WARNING")
    tf.debugging.set_log_device_placement(True)   # zeigt jedes Kernel-Mapping
    assert self._audio_model is None

    all_devices_with_name: list = [
      log_dev
      for log_dev in tf.config.list_logical_devices()
      if self._device_name.lower() in log_dev.name.lower()
    ]
    
    if len(all_devices_with_name) == 0:
      raise Exception("No device found!")
    logical_device = all_devices_with_name[0]
    if len(all_devices_with_name) > 1:
      print(
        f"Multiple devices found: {all_devices_with_name}. "
        f"Using the first one ('{logical_device.name}')."
      )
    physical_devices = tf.config.list_physical_devices('GPU') 
    for gpu_instance in physical_devices: 
      if gpu_instance.name.endswith(logical_device.name[1:]):
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    # tf.config.experimental.set_memory_growth(logical_device, True)
    self._device = logical_device

    self._audio_model = tf.saved_model.load(self._model_path)
    

  @final
  def infer(self, batch: np.ndarray) -> np.ndarray:
    assert self._audio_model is not None
    assert self._device is not None
    # basic_fn = self._audio_model.signatures["basic"]  # oder "basic"

    # keine Retrace-Warnungen, weil wir eine Concrete-Function benutzen
    #prediction = basic_fn(inputs=batch)
    #prediction = prediction["scores"]     
    import tensorflow as tf
    with tf.device(self._device.name):  # type: ignore
      prediction = self._audio_model.basic(batch)["scores"]
    prediction_np = prediction.numpy()
    assert prediction_np.dtype == np.float32
    return prediction_np
