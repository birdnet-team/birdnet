import multiprocessing as mp
from collections.abc import Callable

import numpy as np
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_tensor import (
  AcousticEncodingTensor,
  PrebuiltEncodingTensor,
)
from birdnet.acoustic.inference.core.prediction.prediction_tensor import (
  AcousticPredictionTensor,
  PrebuiltPredictionTensor,
)
from birdnet.acoustic.inference.core.tensor import AcousticTensorBase
from birdnet.utils.helper import UIntCType

pytestmark = pytest.mark.no_tf

N_INPUTS = 3
WIDTH = 4

# Attribute name -> value every array holds in an unprocessable input's rows.
UNPROCESSABLE_VALUES: dict[type, dict[str, int | float | bool]] = {
  AcousticPredictionTensor: {
    "_species_ids": 0,
    "_species_probs": 0.0,
    "_species_masked": True,
  },
  AcousticEncodingTensor: {"_emb": 0, "_emb_masked": True},
}


def _prediction_tensor(max_segment_index: UIntCType) -> AcousticPredictionTensor:
  return AcousticPredictionTensor(
    session_id="test",
    n_inputs=N_INPUTS,
    top_k=WIDTH,
    n_species=10,
    half_precision=False,
    input_indices_dtype=np.dtype(np.uint8),
    segment_indices_dtype=np.dtype(np.uint8),
    max_segment_index=max_segment_index,
  )


def _encoding_tensor(max_segment_index: UIntCType) -> AcousticEncodingTensor:
  return AcousticEncodingTensor(
    session_id="test",
    n_inputs=N_INPUTS,
    emb_dim=WIDTH,
    half_precision=False,
    input_indices_dtype=np.dtype(np.uint8),
    segment_indices_dtype=np.dtype(np.uint8),
    max_segment_index=max_segment_index,
  )


def _write(tensor: AcousticTensorBase, input_idx: int, segment_idx: int) -> None:
  """Write one unmasked, non-zero segment into every array of ``tensor``."""
  inputs = np.array([input_idx], dtype=np.uint8)
  segments = np.array([segment_idx], dtype=np.uint8)
  if isinstance(tensor, AcousticPredictionTensor):
    ids = np.arange(1, WIDTH + 1, dtype=tensor._species_ids.dtype)[None, :]
    scores = np.full((1, WIDTH), 0.5, dtype=tensor._species_probs.dtype)
    tensor.write_block(inputs, segments, ids, scores, np.zeros((1, WIDTH), bool))
  else:
    assert isinstance(tensor, AcousticEncodingTensor)
    tensor.write_block(inputs, segments, np.ones((1, WIDTH), dtype=tensor._emb.dtype))


def _named_arrays(tensor: AcousticTensorBase) -> dict[str, np.ndarray]:
  return {name: getattr(tensor, name) for name in UNPROCESSABLE_VALUES[type(tensor)]}


BUILDERS = [
  pytest.param(_prediction_tensor, id="prediction"),
  pytest.param(_encoding_tensor, id="encoding"),
]


@pytest.mark.parametrize("build", BUILDERS)
def test_unprocessable_inputs_before_set_raises_assertion_error(
  build: Callable[[UIntCType], AcousticTensorBase],
) -> None:
  tensor = build(mp.RawValue("I", 2))

  with pytest.raises(AssertionError):
    tensor.unprocessable_inputs


@pytest.mark.parametrize("build", BUILDERS)
def test_copy_file_slice_returns_independent_copies(
  build: Callable[[UIntCType], AcousticTensorBase],
) -> None:
  tensor = build(mp.RawValue("I", 5))
  _write(tensor, 1, 0)
  _write(tensor, 1, 2)
  named = _named_arrays(tensor)

  copies = tensor.copy_file_slice(1, 3)

  assert len(copies) == len(named)
  for copy, (name, array) in zip(copies, named.items(), strict=True):
    assert copy.shape == (1, 3, WIDTH), name
    assert copy.dtype == array.dtype, name
    assert np.array_equal(copy[0], array[1, :3]), name
    assert not np.shares_memory(copy, array), name


@pytest.mark.parametrize("build", BUILDERS)
def test_arrays_are_the_named_attributes_in_slice_order(
  build: Callable[[UIntCType], AcousticTensorBase],
) -> None:
  tensor = build(mp.RawValue("I", 2))
  named = _named_arrays(tensor)

  arrays = tensor.arrays()  # type: ignore[attr-defined]

  assert len(arrays) == len(named)
  for array, expected in zip(arrays, named.values(), strict=True):
    assert array is expected


@pytest.mark.parametrize("build", BUILDERS)
def test_set_unprocessable_inputs_fills_every_array(
  build: Callable[[UIntCType], AcousticTensorBase],
) -> None:
  tensor = build(mp.RawValue("I", 2))
  _write(tensor, 1, 0)
  _write(tensor, 2, 0)

  tensor.set_unprocessable_inputs({1})

  assert tensor.unprocessable_inputs.tolist() == [1]
  for name, array in _named_arrays(tensor).items():
    expected = UNPROCESSABLE_VALUES[type(tensor)][name]
    assert np.all(array[1] == expected), name
    assert not np.all(array[2, 0] == expected), name


@pytest.mark.parametrize("build", BUILDERS)
def test_resize_keeps_written_rows_of_every_array(
  build: Callable[[UIntCType], AcousticTensorBase],
) -> None:
  max_segment_index = mp.RawValue("I", 2)
  tensor = build(max_segment_index)
  _write(tensor, 1, 1)
  before = {name: array[1, 1].copy() for name, array in _named_arrays(tensor).items()}

  max_segment_index.value = 9
  _write(tensor, 0, 9)

  for name, array in _named_arrays(tensor).items():
    assert array.shape[1] == 10, name
    assert np.array_equal(array[1, 1], before[name]), name


def test_prebuilt_prediction_tensor_wraps_the_given_arrays() -> None:
  ids = np.arange(2 * WIDTH, dtype=np.uint8).reshape(1, 2, WIDTH)
  probs = np.linspace(0.0, 1.0, 2 * WIDTH, dtype=np.float32).reshape(1, 2, WIDTH)
  masked = np.zeros((1, 2, WIDTH), dtype=bool)

  tensor = PrebuiltPredictionTensor(ids, probs, masked)

  assert all(a is b for a, b in zip(tensor.arrays(), (ids, probs, masked), strict=True))
  assert tensor.current_n_segments == 2
  assert tensor.memory_usage_mb == (ids.nbytes + probs.nbytes + masked.nbytes) / 1024**2
  copies = tensor.copy_file_slice(0, 2)
  for copy, array in zip(copies, (ids, probs, masked), strict=True):
    assert copy.shape == (1, 2, WIDTH)
    assert np.array_equal(copy, array)
    assert not np.shares_memory(copy, array)
  with pytest.raises(AssertionError):
    tensor.unprocessable_inputs
  with pytest.raises(NotImplementedError):
    tensor.write_block()


def test_prebuilt_encoding_tensor_wraps_the_given_arrays() -> None:
  emb = np.arange(2 * WIDTH, dtype=np.float32).reshape(1, 2, WIDTH)
  emb_masked = np.zeros((1, 2, WIDTH), dtype=bool)

  tensor = PrebuiltEncodingTensor(emb, emb_masked)

  assert all(a is b for a, b in zip(tensor.arrays(), (emb, emb_masked), strict=True))
  assert tensor.current_n_segments == 2
  assert tensor.memory_usage_mb == (emb.nbytes + emb_masked.nbytes) / 1024**2
  copies = tensor.copy_file_slice(0, 2)
  for copy, array in zip(copies, (emb, emb_masked), strict=True):
    assert copy.shape == (1, 2, WIDTH)
    assert np.array_equal(copy, array)
    assert not np.shares_memory(copy, array)
  with pytest.raises(AssertionError):
    tensor.unprocessable_inputs
  with pytest.raises(NotImplementedError):
    tensor.write_block()
