"""The per-worker estimate: backend runtime, model size and batch."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

import birdnet.acoustic.inference.memory as memory
from birdnet.acoustic.inference.memory import (
  WorkerCost,
  cost_key,
  estimate_worker_bytes,
  model_bytes,
)
from birdnet.core.backends import (
  TF_BACKEND_LIB_ARG,
  OnnxBackend,
  PBBackend,
  TFBackend,
  TorchBackend,
)
from birdnet_tests.fake_acoustic_backend import FakeAcousticBackend

pytestmark = pytest.mark.no_tf

_MIB = 2**20


class _TF(TFBackend):
  pass


class _PB(PBBackend):
  pass


class _Onnx(OnnxBackend):
  pass


class _Torch(TorchBackend):
  pass


@pytest.mark.parametrize(
  ("backend", "kwargs", "key"),
  [
    (_TF, {TF_BACKEND_LIB_ARG: "tflite"}, "tflite"),
    (_TF, {TF_BACKEND_LIB_ARG: "litert"}, "litert"),
    (_PB, {}, "pb"),
    (_Onnx, {}, "onnx"),
    (_Torch, {}, "pt"),
    (FakeAcousticBackend, {}, "other"),
  ],
)
def test_cost_key_per_backend(backend: type, kwargs: dict, key: str) -> None:
  assert cost_key(backend, kwargs) == key  # type: ignore[arg-type]


def test_litert_costs_less_runtime_than_tensorflow() -> None:
  assert (
    memory._WORKER_COSTS["litert"].runtime_bytes
    < memory._WORKER_COSTS["tflite"].runtime_bytes
  )


@pytest.mark.parametrize(
  ("key", "model_mib", "measured_mib"),
  [
    # Max RssAnon of one of 12 workers at batch size 1 (see _WORKER_COSTS).
    ("tflite", 49, 304),
    ("tflite", 328, 579),
    ("litert", 49, 99),
    ("litert", 328, 374),
    ("onnx", 329, 544),
  ],
)
def test_the_estimate_is_at_most_five_percent_under_a_measured_worker(
  key: str, model_mib: int, measured_mib: int
) -> None:
  estimate = memory._WORKER_COSTS[key].estimate(model_mib * _MIB, batch_size=1)
  assert estimate >= 0.95 * measured_mib * _MIB


@pytest.mark.parametrize(
  ("library", "model_mib", "measured_mib"),
  [
    # Unique set per worker, 12 workers at batch size 1, Windows 11.
    ("tflite", 49, 340),
    ("tflite", 328, 900),
    ("litert", 49, 100),
    ("litert", 328, 380),
  ],
)
def test_on_windows_the_estimate_covers_a_measured_worker_too(
  monkeypatch: pytest.MonkeyPatch, library: str, model_mib: int, measured_mib: int
) -> None:
  monkeypatch.setattr(sys, "platform", "win32")

  estimate = estimate_worker_bytes(
    _TF, {TF_BACKEND_LIB_ARG: library}, model_mib * _MIB, batch_size=1
  )

  assert 0.95 * measured_mib * _MIB <= estimate <= 1.4 * measured_mib * _MIB


def test_only_tensorflows_interpreter_on_windows_holds_the_file_twice(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def estimate(platform: str, library: str) -> int:
    monkeypatch.setattr(sys, "platform", platform)
    return estimate_worker_bytes(
      _TF, {TF_BACKEND_LIB_ARG: library}, 328 * _MIB, batch_size=1
    )

  assert estimate("win32", "tflite") - estimate("linux", "tflite") == 328 * _MIB
  assert estimate("win32", "litert") == estimate("linux", "litert")
  assert estimate("darwin", "tflite") == estimate("linux", "tflite")


def _fp32_backend(version: str) -> type:
  if version == "2.4":
    from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4

    return AcousticTFBackendFP32V2_4
  from birdnet.acoustic.models.v3_0.tf import AcousticTFBackendFP32V3_0

  return AcousticTFBackendFP32V3_0


@pytest.mark.parametrize(
  ("version", "library", "model_mib", "measured_mib"),
  [
    # Largest footprint of four workers without XNNPACK, MacBook Air M1.
    ("2.4", "tflite", 49, 257),
    ("3.0", "tflite", 328, 310),
    ("2.4", "litert", 49, 58),
    ("3.0", "litert", 328, 117),
  ],
)
def test_on_apple_silicon_a_worker_is_planned_without_xnnpacks_weights(
  monkeypatch: pytest.MonkeyPatch,
  version: str,
  library: str,
  model_mib: int,
  measured_mib: int,
) -> None:
  backend, kwargs = _fp32_backend(version), {TF_BACKEND_LIB_ARG: library}

  def estimate(system: str, machine: str) -> int:
    monkeypatch.setattr(sys, "platform", system.lower())
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "machine", lambda: machine)
    return estimate_worker_bytes(backend, kwargs, model_mib * _MIB, batch_size=1)

  on_m1 = estimate("Darwin", "arm64")

  assert 0.95 * measured_mib * _MIB <= on_m1 <= 1.5 * measured_mib * _MIB
  assert estimate("Linux", "x86_64") - on_m1 == model_mib * _MIB
  assert estimate("Darwin", "x86_64") - on_m1 == model_mib * _MIB


def test_on_apple_silicon_the_plan_of_other_precisions_is_left_alone(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from birdnet.acoustic.models.v2_4.tf import (
    AcousticTFBackendFP16V2_4,
    AcousticTFBackendInt8V2_4,
  )
  from birdnet.acoustic.models.v3_0.tf import AcousticTFBackendFP16V3_0

  kwargs = {TF_BACKEND_LIB_ARG: "tflite"}
  monkeypatch.setattr(sys, "platform", "darwin")
  monkeypatch.setattr(platform, "system", lambda: "Darwin")
  monkeypatch.setattr(platform, "machine", lambda: "arm64")

  for backend in (
    AcousticTFBackendFP16V2_4,
    AcousticTFBackendInt8V2_4,
    AcousticTFBackendFP16V3_0,
  ):
    assert estimate_worker_bytes(
      backend, kwargs, 100 * _MIB, 1
    ) == memory._WORKER_COSTS["tflite"].estimate(100 * _MIB, 1)


def test_estimate_is_runtime_plus_model_plus_batch() -> None:
  cost = WorkerCost(100 * _MIB, 1.5, 10 * _MIB)
  assert cost.estimate(model_bytes=200 * _MIB, batch_size=4) == 440 * _MIB


def test_a_batch_run_row_by_row_adds_the_outputs_of_its_other_rows() -> None:
  cost = WorkerCost(100 * _MIB, 1.5, 10 * _MIB)

  one = cost.estimate(model_bytes=200 * _MIB, batch_size=1, row_by_row=True)
  four = cost.estimate(model_bytes=200 * _MIB, batch_size=4, row_by_row=True)

  assert one == 410 * _MIB
  assert cost.estimate(model_bytes=200 * _MIB, batch_size=1) == one
  assert four == 410 * _MIB + 3 * memory._ROW_OUTPUT_BYTES


def test_model_bytes_of_a_file_and_of_a_directory(tmp_path: Path) -> None:
  single = tmp_path / "model.tflite"
  single.write_bytes(b"x" * 1000)
  assert model_bytes(single) == 1000
  saved = tmp_path / "saved_model"
  (saved / "variables").mkdir(parents=True)
  (saved / "saved_model.pb").write_bytes(b"x" * 300)
  (saved / "variables" / "variables.data").write_bytes(b"x" * 700)
  assert model_bytes(saved) == 1000
  assert model_bytes(tmp_path / "missing") == 0


def test_estimate_uses_the_backends_row() -> None:
  onnx = memory._WORKER_COSTS["onnx"]
  assert estimate_worker_bytes(_Onnx, {}, 100 * _MIB, batch_size=2) == int(
    onnx.runtime_bytes + onnx.model_factor * 100 * _MIB + 2 * onnx.per_segment_bytes
  )


def test_a_ring_slot_is_four_bytes_per_sample(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  # 1 GiB after the reserve (256 MiB plus the producer's 1.5 slots) and the
  # 0.9 planning fraction is 683 MiB; with no worker memory at all, 2 slots of
  # batch 16 x 100 000 samples x 4 B each (12.2 MiB per worker) fit 55 times,
  # not the 225 that 1 byte per sample would give.
  monkeypatch.setattr(memory, "physical_cores", lambda: 1000)
  monkeypatch.setattr(memory, "host_memory_available", lambda: 1024 * _MIB)
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: None)
  monkeypatch.setattr(memory, "estimate_worker_bytes", lambda *_a, **_k: 0)
  n = memory.default_n_workers(
    backend_type=FakeAcousticBackend,  # type: ignore[arg-type]
    backend_kwargs={},
    model_path=tmp_path / "none",
    n_producers=1,
    batch_size=16,
    prefetch_ratio=1,
    segment_samples=100_000,
  )
  assert n == 55
