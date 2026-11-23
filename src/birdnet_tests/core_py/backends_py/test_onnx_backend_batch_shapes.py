"""Which input shapes an ONNX backend hands to its session.

On a GPU every change of the input shape is expensive, so a batch shorter than
the largest one so far is filled up to it with copies of its last segment, and
the padding rows are dropped. On one CPU thread a batch gains nothing and costs
cache, so its rows run one by one; with several threads it runs as it is.
Pinned on a session stand-in.
"""

from pathlib import Path

import numpy as np
import pytest

from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0

pytestmark = pytest.mark.no_tf


class _Session:
  def __init__(self) -> None:
    self.shapes: list[tuple[int, ...]] = []
    self.batches: list[np.ndarray] = []

  def run(self, output_names: list[str], feed: dict[str, np.ndarray]) -> list:
    [batch] = feed.values()
    self.shapes.append(batch.shape)
    self.batches.append(batch.copy())
    # One row per input row, so a row of padding is told from a real one.
    return [batch[:, :2] * 2]


def _backend(
  device: str, cpu_threads: int | None = None
) -> tuple[AcousticOnnxBackendFP32V3_0, _Session]:
  backend = AcousticOnnxBackendFP32V3_0(
    model_path=Path("does-not-exist.onnx"),
    device_name=device,
    half_precision=False,
    cpu_threads=cpu_threads,
  )
  session = _Session()
  backend._session = session  # type: ignore[assignment]
  backend._input_name = "input"
  backend._output_names = ["predictions", "embeddings"]
  return backend, session


_FULL = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
_SHORT = _FULL[:2] * 10


def test_gpu_runs_a_short_batch_at_the_largest_shape_so_far() -> None:
  backend, session = _backend("GPU")

  backend.predict(_FULL)
  result = backend.predict(_SHORT)

  assert session.shapes == [(4, 3), (4, 3)]
  np.testing.assert_array_equal(result, _SHORT[:, :2] * 2)


def test_gpu_pads_with_the_last_segment_not_with_silence() -> None:
  backend, session = _backend("GPU")

  backend.predict(_FULL)
  backend.predict(_SHORT)

  np.testing.assert_array_equal(session.batches[1][2:], [_SHORT[-1], _SHORT[-1]])


def test_gpu_runs_embeddings_of_a_short_batch_padded_too() -> None:
  backend, session = _backend("GPU:1")

  backend.encode(_FULL)
  result = backend.encode(_SHORT)

  assert session.shapes == [(4, 3), (4, 3)]
  assert result.shape == (2, 2)


def test_gpu_runs_a_batch_larger_than_any_before_as_it_is() -> None:
  backend, session = _backend("GPU")

  backend.predict(_SHORT)
  result = backend.predict(_FULL)

  assert session.shapes == [(2, 3), (4, 3)]
  np.testing.assert_array_equal(result, _FULL[:, :2] * 2)


def test_one_cpu_thread_runs_the_rows_one_by_one() -> None:
  backend, session = _backend("CPU", cpu_threads=1)

  result = backend.predict(_FULL)

  assert session.shapes == [(1, 3)] * 4
  np.testing.assert_array_equal(result, _FULL[:, :2] * 2)


def test_one_gpu_keeps_the_batch_whatever_the_cpu_threads() -> None:
  backend, session = _backend("GPU", cpu_threads=1)

  backend.predict(_FULL)

  assert session.shapes == [(4, 3)]


def test_outside_the_pipeline_a_batch_runs_at_its_own_shape() -> None:
  """No thread count was handed in: the validity check and direct callers."""
  backend, session = _backend("CPU")

  backend.predict(_FULL)
  result = backend.predict(_SHORT)

  assert session.shapes == [(4, 3), (2, 3)]
  np.testing.assert_array_equal(result, _SHORT[:, :2] * 2)


def test_unload_forgets_the_largest_batch() -> None:
  backend, session = _backend("GPU")
  backend.predict(_FULL)

  backend.unload()
  backend._session = session  # type: ignore[assignment]
  backend._input_name = "input"
  backend._output_names = ["predictions", "embeddings"]
  backend.predict(_SHORT)

  assert session.shapes == [(4, 3), (2, 3)]


def test_several_cpu_threads_run_a_batch_as_it_is() -> None:
  backend, session = _backend("CPU", cpu_threads=3)

  result = backend.predict(_FULL)

  assert session.shapes == [(4, 3)]
  np.testing.assert_array_equal(result, _FULL[:, :2] * 2)
