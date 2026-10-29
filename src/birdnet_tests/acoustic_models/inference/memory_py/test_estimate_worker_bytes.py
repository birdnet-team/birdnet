"""The per-worker estimate: backend runtime, model size and batch."""

from __future__ import annotations

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


def test_estimate_is_runtime_plus_model_plus_batch() -> None:
  cost = WorkerCost(100 * _MIB, 1.5, 10 * _MIB)
  assert cost.estimate(model_bytes=200 * _MIB, batch_size=4) == 440 * _MIB


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
  # 1 GiB after the reserve and the 0.9 planning fraction is 691 MiB; with no
  # worker memory at all, 2 slots of batch 16 x 100 000 samples x 4 B each
  # (12.2 MiB per worker) fit 56 times, not the 226 that 1 byte per sample
  # would give.
  monkeypatch.setattr(memory, "physical_cores", lambda: 1000)
  monkeypatch.setattr(memory, "host_memory_available", lambda: 1024 * _MIB)
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: None)
  monkeypatch.setattr(memory, "estimate_worker_bytes", lambda *_a, **_k: 0)
  n = memory.default_n_workers(
    backend_type=FakeAcousticBackend,  # type: ignore[arg-type]
    backend_kwargs={},
    model_path=tmp_path / "none",
    batch_size=16,
    prefetch_ratio=1,
    segment_samples=100_000,
  )
  assert n == 56
