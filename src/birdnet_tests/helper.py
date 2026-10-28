import contextlib
import ctypes
import multiprocessing as mp
import os
import sys
import threading
import time
import zipfile
from collections.abc import Callable, Generator, Iterable
from multiprocessing import get_all_start_methods, set_start_method
from pathlib import Path
from typing import IO

import numpy as np
import psutil
import pytest

from birdnet.acoustic.inference.core.encoding.encoding_result import (
  AcousticEncodingResultBase,
)
from birdnet.acoustic.inference.core.prediction.prediction_result import (
  AcousticPredictionResultBase,
)
from birdnet.core.backends import (
  _get_major_minor_version,
  check_is_intel_macos,
  litert_installed,
  onnxruntime_installed,
  tf_installed,
  torch_installed,
)
from birdnet.core.base import (
  NP_MODEL_PATH_KEY,
  NP_MODEL_PRECISION_KEY,
  NP_MODEL_VERSION_KEY,
  ResultBase,
)
from birdnet.utils import downloads


def check_is_python_312() -> bool:
  return sys.version_info.major == 3 and sys.version_info.minor == 12


def create_zero_len_wav(f: IO[bytes]) -> None:
  # 44-Byte WAV Header
  f.write(
    b"RIFF"
    + (36).to_bytes(4, "little")
    + b"WAVEfmt "
    + (16).to_bytes(4, "little")  # fmt chunk length
    + (1).to_bytes(2, "little")  # PCM
    + (1).to_bytes(2, "little")  # channels
    + (44100).to_bytes(4, "little")  # sample rate
    + (88200).to_bytes(4, "little")  # byte rate
    + (2).to_bytes(2, "little")  # block align
    + (16).to_bytes(2, "little")  # bits per sample
    + b"data"
    + (0).to_bytes(4, "little")  # data size = 0
  )


def assert_encoding_result_is_equal(
  a: AcousticEncodingResultBase, b: AcousticEncodingResultBase
) -> None:
  assert isinstance(a, AcousticEncodingResultBase), (
    "a must be AcousticEncodingResultBase"
  )
  assert isinstance(b, AcousticEncodingResultBase), (
    "b must be AcousticEncodingResultBase"
  )

  np.testing.assert_array_equal(
    a.segment_duration_s, b.segment_duration_s, err_msg="segment_duration_s"
  )
  np.testing.assert_array_equal(
    a.overlap_duration_s, b.overlap_duration_s, err_msg="overlap_duration_s"
  )
  np.testing.assert_array_equal(a.inputs, b.inputs, err_msg="inputs")
  np.testing.assert_array_equal(
    a.input_durations, b.input_durations, err_msg="input_durations"
  )
  np.testing.assert_array_equal(a.embeddings, b.embeddings, err_msg="embeddings")
  np.testing.assert_array_equal(
    a.embeddings_masked, b.embeddings_masked, err_msg="embeddings_masked"
  )


def assert_prediction_result_is_equal(
  a: AcousticPredictionResultBase, b: AcousticPredictionResultBase
) -> None:
  assert isinstance(a, AcousticPredictionResultBase), (
    "a must be AcousticPredictionResultBase"
  )
  assert isinstance(b, AcousticPredictionResultBase), (
    "b must be AcousticPredictionResultBase"
  )

  np.testing.assert_array_equal(
    a.segment_duration_s, b.segment_duration_s, err_msg="segment_duration_s"
  )
  np.testing.assert_array_equal(
    a.overlap_duration_s, b.overlap_duration_s, err_msg="overlap_duration_s"
  )
  np.testing.assert_array_equal(a.inputs, b.inputs, err_msg="inputs")
  np.testing.assert_array_equal(a.species_list, b.species_list, err_msg="species_list")
  np.testing.assert_array_equal(
    a.input_durations, b.input_durations, err_msg="input_durations"
  )

  # Sort species probabilities by species IDs before comparison
  sort_idx_a = np.argsort(a.species_ids, axis=-1)
  sort_idx_b = np.argsort(b.species_ids, axis=-1)

  sorted_ids_a = np.take_along_axis(a.species_ids, sort_idx_a, axis=-1)
  sorted_ids_b = np.take_along_axis(b.species_ids, sort_idx_b, axis=-1)

  # order may differ due to different top-k selection, but ids must be the same
  np.testing.assert_array_equal(sorted_ids_a, sorted_ids_b, err_msg="species_ids")

  sorted_probs_a = np.take_along_axis(a.species_probs, sort_idx_a, axis=-1)
  sorted_probs_b = np.take_along_axis(b.species_probs, sort_idx_b, axis=-1)
  np.testing.assert_array_equal(sorted_probs_a, sorted_probs_b, err_msg="species_probs")

  sorted_masks_a = np.take_along_axis(a.species_masked, sort_idx_a, axis=-1)
  sorted_masks_b = np.take_along_axis(b.species_masked, sort_idx_b, axis=-1)
  np.testing.assert_array_equal(
    sorted_masks_a, sorted_masks_b, err_msg="species_masked"
  )


def assert_encoding_result_is_close(
  a: AcousticEncodingResultBase,
  b: AcousticEncodingResultBase,
  max_abs_diff: float | None,
  mean_abs_diff: float | None = None,
) -> None:
  assert isinstance(a, AcousticEncodingResultBase)
  assert isinstance(b, AcousticEncodingResultBase)

  np.testing.assert_array_equal(
    a.segment_duration_s, b.segment_duration_s, err_msg="segment_duration_s"
  )
  np.testing.assert_array_equal(
    a.overlap_duration_s, b.overlap_duration_s, err_msg="overlap_duration_s"
  )
  np.testing.assert_array_equal(a.inputs, b.inputs, err_msg="inputs")
  np.testing.assert_array_equal(
    a.input_durations, b.input_durations, err_msg="input_durations"
  )

  np.testing.assert_array_equal(
    a.embeddings.shape, b.embeddings.shape, err_msg="embeddings.shape"
  )
  _assert_array_is_close(
    a.embeddings,
    b.embeddings,
    max_abs_diff=max_abs_diff,
    mean_abs_diff=mean_abs_diff,
    label="embeddings",
  )


def assert_prediction_result_is_close(
  a: AcousticPredictionResultBase,
  b: AcousticPredictionResultBase,
  max_abs_diff: float | None,
  mean_abs_diff: float | None = None,
) -> None:
  assert isinstance(a, AcousticPredictionResultBase), (
    "a must be AcousticPredictionResultBase"
  )
  assert isinstance(b, AcousticPredictionResultBase), (
    "b must be AcousticPredictionResultBase"
  )

  np.testing.assert_array_equal(
    a.segment_duration_s, b.segment_duration_s, err_msg="segment_duration_s"
  )
  np.testing.assert_array_equal(
    a.overlap_duration_s, b.overlap_duration_s, err_msg="overlap_duration_s"
  )
  np.testing.assert_array_equal(a.inputs, b.inputs, err_msg="inputs")
  np.testing.assert_array_equal(a.species_list, b.species_list, err_msg="species_list")
  np.testing.assert_array_equal(
    a.input_durations, b.input_durations, err_msg="input_durations"
  )

  # Sort species probabilities by species IDs before comparison
  sort_idx_a = np.argsort(a.species_ids, axis=-1)
  sort_idx_b = np.argsort(b.species_ids, axis=-1)

  sorted_ids_a = np.take_along_axis(a.species_ids, sort_idx_a, axis=-1)
  sorted_ids_b = np.take_along_axis(b.species_ids, sort_idx_b, axis=-1)

  # order may differ due to different top-k selection, but ids must be the same
  np.testing.assert_array_equal(sorted_ids_a, sorted_ids_b, err_msg="species_ids")

  sorted_masks_a = np.take_along_axis(a.species_masked, sort_idx_a, axis=-1)
  sorted_masks_b = np.take_along_axis(b.species_masked, sort_idx_b, axis=-1)
  np.testing.assert_array_equal(
    sorted_masks_a, sorted_masks_b, err_msg="species_masked"
  )

  sorted_probs_a = np.take_along_axis(a.species_probs, sort_idx_a, axis=-1)
  sorted_probs_b = np.take_along_axis(b.species_probs, sort_idx_b, axis=-1)

  np.testing.assert_array_equal(
    sorted_probs_a.shape, sorted_probs_b.shape, err_msg="species_probs.shape"
  )
  _assert_array_is_close(
    sorted_probs_a,
    sorted_probs_b,
    max_abs_diff=max_abs_diff,
    mean_abs_diff=mean_abs_diff,
    label="species_probs",
  )


def _assert_array_is_close(
  a: np.ndarray,
  b: np.ndarray,
  *,
  max_abs_diff: float | None,
  mean_abs_diff: float | None,
  label: str,
) -> None:
  if max_abs_diff is None and mean_abs_diff is None:
    raise ValueError("At least one difference threshold must be provided.")

  diff = np.abs(a - b)

  if max_abs_diff is not None:
    max_abs = np.max(diff)
    assert max_abs <= max_abs_diff, (
      f"{label} max absolute difference {max_abs} exceeds threshold {max_abs_diff}"
    )

  if mean_abs_diff is not None:
    mean_abs = np.mean(diff)
    assert mean_abs <= mean_abs_diff, (
      f"{label} mean absolute difference {mean_abs} exceeds threshold {mean_abs_diff}"
    )


def _check_tf_gpu() -> bool:
  try:
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    return len(devices) > 0
  except Exception:
    return False


def tensorflow_gpu_available() -> bool:
  ctx = mp.get_context()
  with ctx.Pool(1) as pool:
    result = pool.apply(_check_tf_gpu)
  return result


def ensure_gpu_or_skip() -> None:
  if not tensorflow_gpu_available():
    pytest.skip("GPU not available")


def ensure_not_intel_macos_or_skip() -> None:
  if check_is_intel_macos():
    pytest.skip("Test not supported on Intel macOS systems")


def ensure_v3_0_torch_backend_or_skip() -> None:
  # torch 2.2.2 is the last release shipping x86 macOS wheels and is too old for the
  # v3.0 TorchScript model: loading it succeeds, but the inference workers then die and
  # the session is cancelled. The onnx backend works on the same machines.
  if check_is_intel_macos():
    pytest.skip("Acoustic model v3.0 needs a newer torch than Intel macOS provides")


def ensure_litert_or_skip() -> None:
  if not litert_installed():
    pytest.skip("litert library is not available")


def ensure_torch_or_skip() -> None:
  if not torch_installed():
    pytest.skip("PyTorch runtime is not available")


def ensure_validity_check_or_skip() -> None:
  # Opening a model in the validity-check subprocess times out there, see
  # model_loader_py/acoustic/test_acoustic_load_custom.py.
  if check_is_python_312() and check_is_intel_macos():
    pytest.skip("The model validity check times out on Intel macOS, Python 3.12")


def ensure_onnxruntime_or_skip() -> None:
  if not onnxruntime_installed():
    pytest.skip("ONNX Runtime is not available")


def geo_v3_0_litert_not_supported_skip() -> None:
  pytest.skip("Geo model v3.0 TF backend is not supported with ai_edge_litert yet")


def _tf_version_or_skip() -> str:
  if not tf_installed():
    pytest.skip("TensorFlow is not installed (pip install birdnet[tf])")
  import tensorflow as tf

  version: str = tf.__version__
  return version


def _tf_major_minor_or_skip() -> tuple[int, int]:
  version = _tf_version_or_skip()
  try:
    return _get_major_minor_version(version)
  except RuntimeError:
    pytest.skip(f"Unsupported TensorFlow version string for this test: {version!r}")


def ensure_tf_2_19_or_2_18() -> None:
  """Skip where the geo v3.0 tf backend's own version check rejects TensorFlow."""
  _tf_version_or_skip()
  from birdnet.geo.models.v3_0.tf import _check_tf_version_for_v3_0

  try:
    _check_tf_version_for_v3_0()
  except RuntimeError as e:
    pytest.skip(str(e))


def ensure_tf_2_18_or_skip() -> None:
  """Skip where the TFLite runtime is too old to even parse the current models.

  The macOS Intel wheels are pinned to TensorFlow <2.17, whose interpreter cannot
  open the geo v3.0 INT8 export ("Didn't find op for builtin opcode
  'FULLY_CONNECTED' version '12'"). The geo v3.0 TF backend requires 2.18/2.19
  anyway, so there is nothing to verify on such a runtime.
  """
  if _tf_major_minor_or_skip() < (2, 18):
    pytest.skip("TensorFlow >= 2.18 is required to load the geo v3.0 models")


def ensure_tf_2_20_or_skip() -> None:
  if _tf_major_minor_or_skip() < (2, 20):
    pytest.skip("TensorFlow >= 2.20 is required for this test")


def ensure_not_mac_or_skip() -> None:
  import platform

  if platform.system() == "Darwin":
    pytest.skip("Test not supported on macOS systems")


def use_forkserver_or_skip() -> None:
  if "forkserver" in get_all_start_methods():
    set_start_method("forkserver", force=True)
  else:
    pytest.skip("forkserver start method not available on this platform")


def use_fork_or_skip() -> None:
  if "fork" in get_all_start_methods():
    set_start_method("fork", force=True)
  else:
    pytest.skip("fork start method not available on this platform")


def use_spawn_or_skip() -> None:
  if "spawn" in get_all_start_methods():
    set_start_method("spawn", force=True)
  else:
    pytest.skip("spawn start method not available on this platform")


@contextlib.contextmanager
def memory_monitor() -> Generator[Callable, None, None]:
  """Context manager für Memory-Monitoring."""
  process = psutil.Process(os.getpid())
  memory_before = process.memory_full_info().uss
  max_memory = ctypes.c_float(memory_before)
  stop_event = threading.Event()

  def monitor_worker() -> None:
    while not stop_event.is_set():
      try:
        current_memory = process.memory_full_info().uss
        if current_memory > max_memory.value:
          max_memory.value = current_memory
        time.sleep(0.05)
      except (psutil.NoSuchProcess, psutil.AccessDenied):
        break

  monitor_thread = threading.Thread(
    target=monitor_worker, daemon=True, name="MemoryMonitorThread"
  )
  monitor_thread.start()

  def get_memory_delta() -> float:
    return (max_memory.value - memory_before) / 1024**2

  try:
    yield get_memory_delta
  finally:
    stop_event.set()
    monitor_thread.join(timeout=1.0)


@contextlib.contextmanager
def duration_counter() -> Generator[Callable, None, None]:
  """Context manager to measure duration of a code block."""
  start = time.perf_counter()

  def get_duration() -> float:
    end = time.perf_counter()
    return end - start

  try:
    yield get_duration
  finally:
    pass  # No cleanup needed


def get_max_absolute_tolerance(a: np.ndarray, b: np.ndarray) -> float:
  a = np.asarray(a)
  b = np.asarray(b)

  diff = np.abs(a - b)
  maximum = np.max(diff)

  return maximum


def get_max_absolute_tolerance_threshold(
  a: np.ndarray, b: np.ndarray, threshold: float = 0.1
) -> float:
  a = np.asarray(a)
  b = np.asarray(b)

  mask = a > threshold
  if not np.any(mask):
    return 0.0

  diff = np.abs(a - b)
  masked_diff = diff[mask]

  mean = np.mean(masked_diff)

  return mean


def estimate_best_rtol_atol(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
  a = np.asarray(a)
  b = np.asarray(b)

  diff = np.abs(a - b)
  denom = np.abs(b)

  with np.errstate(divide="ignore", invalid="ignore"):
    rel = np.where(denom == 0, np.inf, diff / denom)

  max_rel = np.max(rel)
  max_abs = np.max(diff)

  return max_rel, max_abs


def get_max_relative_tolerance(a: np.ndarray, b: np.ndarray) -> float:
  a = np.asarray(a)
  b = np.asarray(b)

  diff = np.abs(a - b)
  denom = np.abs(b)

  # Sonderfall: b == 0 → relative Fehler ist nur sinnvoll, wenn diff ebenfalls 0 ist
  with np.errstate(divide="ignore", invalid="ignore"):
    rel = np.where(denom == 0, np.inf, diff / denom)

  return np.max(rel)


def scores_at_sigmoid_sensitivity(scores: np.ndarray, sensitivity: float) -> np.ndarray:
  """The pipeline's scaled sigmoid on the logits behind a model's sigmoid scores.

  An independent float64 reference for what the prediction worker does with a
  sigmoid-in-graph model at a non-default sensitivity. It shares no code with
  the library's helpers, so a recovery that does nothing is caught.
  """
  p = scores.astype(np.float64)
  with np.errstate(divide="ignore"):
    logits = np.log(p) - np.log(1.0 - p)
  shifted = np.clip(logits + (sensitivity - 1.0) * 10.0, -15.0, 15.0)
  return 1.0 / (1.0 + np.exp(-shifted))


def worst_decimal_precision(a: np.ndarray, b: np.ndarray, max_decimals: int = 8) -> int:
  diff = np.abs(a - b)
  max_diff = np.max(diff)

  for decimals in range(max_decimals, 0, -1):
    if max_diff < 10 ** (-decimals):
      return decimals
  return 0


def create_fake_saved_model(model_dir: Path) -> None:
  """Write the file layout `check_protobuf_model_files_exist` looks for."""
  (model_dir / "variables").mkdir(parents=True, exist_ok=True)
  (model_dir / "saved_model.pb").write_bytes(b"")
  (model_dir / "variables" / "variables.data-00000-of-00001").write_bytes(b"")
  (model_dir / "variables" / "variables.index").write_bytes(b"")


def fake_labels(n_classes: int) -> list[str]:
  """A label list of the length a model's downloader accepts."""
  return [f"Genus species{index}_Species {index}" for index in range(n_classes)]


def create_fake_lang_dir(
  lang_dir: Path, languages: Iterable[str], n_classes: int
) -> None:
  lang_dir.mkdir(parents=True, exist_ok=True)
  content = "\n".join(fake_labels(n_classes)) + "\n"
  for lang in languages:
    (lang_dir / f"{lang}.txt").write_text(content, encoding="utf-8")


def saved_model_members(prefix: str) -> dict[str, bytes]:
  """Zip members of the layout `check_protobuf_model_files_exist` looks for."""
  return {
    f"{prefix}saved_model.pb": b"",
    f"{prefix}variables/variables.data-00000-of-00001": b"",
    f"{prefix}variables/variables.index": b"",
  }


def label_members(
  prefix: str, languages: Iterable[str], n_classes: int
) -> dict[str, bytes]:
  content = ("\n".join(fake_labels(n_classes)) + "\n").encode()
  return {f"{prefix}{lang}.txt": content for lang in languages}


def fake_zip_downloads(
  monkeypatch: pytest.MonkeyPatch,
  archives: dict[str, dict[str, bytes]],
  *,
  held_lock: Path | None = None,
) -> list[str]:
  """Serve `archives` (URL to member path to content) instead of downloading.

  Returns the list the requested URLs are appended to. With `held_lock`, every
  download asserts that this lock directory is held while it runs.
  """
  calls: list[str] = []

  def fake(
    url: str,
    file_path: Path,
    *,
    download_size: int | None = None,
    description: str | None = None,
  ) -> int:
    calls.append(url)
    if held_lock is not None:
      assert (held_lock / "owner").is_file(), f"downloading without {held_lock}"
    with zipfile.ZipFile(file_path, "w") as archive:
      for member, content in archives[url].items():
        archive.writestr(member, content)
    return file_path.stat().st_size

  monkeypatch.setattr(downloads, "download_file_tqdm", fake)
  return calls


def saved_arrays(result: ResultBase) -> dict[str, np.ndarray]:
  """Every array ``ResultBase.save`` writes, as the result holds it now."""
  return {
    NP_MODEL_PATH_KEY: result._model_path,
    NP_MODEL_VERSION_KEY: result._model_version,
    NP_MODEL_PRECISION_KEY: result._model_precision,
  } | result._get_extra_save_data()
