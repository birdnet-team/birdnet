"""The pb backend's GPU lookup, with TensorFlow's answer about its GPUs faked.

A TensorFlow without a GPU, or one that cannot load its CUDA libraries, lists
none; a worker that asks for one then has to say so.
"""

from types import SimpleNamespace

import pytest

from birdnet.core.backends import set_gpu_device_tf


def test_no_gpu_is_reported_with_the_way_out(monkeypatch: pytest.MonkeyPatch) -> None:
  import tensorflow as tf

  # The lookup pins the device through the environment; restored afterwards.
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [])

  with pytest.raises(ValueError, match=r"No GPU found!.*birdnet\[and-cuda\]"):
    set_gpu_device_tf("GPU", memory_growth=True)


def test_memory_growth_is_set_before_the_logical_devices_are_listed(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import tensorflow as tf

  gpu = SimpleNamespace(name="/physical_device:GPU:0")
  calls: list[str] = []

  def set_memory_growth(device: object, enable: bool) -> None:
    assert device is gpu
    assert enable
    calls.append("memory growth")

  def list_logical_devices(kind: str) -> list[SimpleNamespace]:
    calls.append("logical devices")
    return [SimpleNamespace(name="/device:GPU:0")]

  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [gpu])
  monkeypatch.setattr(tf.config.experimental, "set_memory_growth", set_memory_growth)
  monkeypatch.setattr(tf.config, "list_logical_devices", list_logical_devices)

  assert set_gpu_device_tf("GPU", memory_growth=True) == "/device:GPU:0"
  # TensorFlow refuses the setting once it has initialized its devices.
  assert calls == ["memory growth", "logical devices"]
