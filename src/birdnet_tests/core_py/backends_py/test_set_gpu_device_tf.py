"""The pb backend's GPU lookup, with TensorFlow's answer about its GPUs faked.

A TensorFlow without a GPU, or one that cannot load its CUDA libraries, lists
none; a worker that asks for one then has to say so.
"""

import pytest

from birdnet.core.backends import set_gpu_device_tf


def test_no_gpu_is_reported_with_the_way_out(monkeypatch: pytest.MonkeyPatch) -> None:
  import tensorflow as tf

  # The lookup pins the device through the environment; restored afterwards.
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
  monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind: [])

  with pytest.raises(ValueError, match=r"No GPU found!.*birdnet\[and-cuda\]"):
    set_gpu_device_tf("GPU", memory_growth=True)
