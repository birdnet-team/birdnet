"""``n_workers=None`` fits the memory available; an explicit count is untouched."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import psutil
import pytest

import birdnet.acoustic.inference.memory as memory
from birdnet_tests.fake_acoustic_backend import fake_predict_session

pytestmark = pytest.mark.no_tf

_MIB = 2**20


@pytest.fixture(autouse=True)
def _package_logger_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
  # The package logger does not propagate; before pytest 9.1 caplog only saw
  # records that reached the root logger.
  monkeypatch.setattr(logging.getLogger("birdnet"), "propagate", True)


def _simulate(monkeypatch: pytest.MonkeyPatch, available_mib: int) -> None:
  monkeypatch.setattr(memory, "physical_cores", lambda: 32)
  monkeypatch.setattr(memory, "host_memory_available", lambda: available_mib * _MIB)
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: None)
  monkeypatch.setattr(memory, "estimate_worker_bytes", lambda *_a, **_k: 200 * _MIB)


def test_the_default_shrinks_under_a_simulated_limit(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  # (1 GiB - 128 MiB parent - 129 MiB producer with its slots) * 0.9 = 690 MiB
  # holds three workers of ~200 MiB (the fake backend's two ring slots add
  # 1.1 MiB).
  _simulate(monkeypatch, 1024)
  with caplog.at_level(logging.INFO, logger="birdnet"):
    session = fake_predict_session(tmp_path, n_workers=None)
  assert session._conf.processing_conf.workers == 3
  decisions = [
    r for r in caplog.records if "n_workers defaulted to 3" in r.getMessage()
  ]
  assert len(decisions) == 1
  assert decisions[0].levelno == logging.INFO
  assert "32 would need" in decisions[0].getMessage()
  assert "after 256 MiB for the parent and 1 producer" in decisions[0].getMessage()


def test_the_sessions_producers_are_kept_back(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  # A second producer leaves (1024 - 385.6) * 0.9 = 574 MiB, two workers.
  _simulate(monkeypatch, 1024)
  with caplog.at_level(logging.INFO, logger="birdnet"):
    session = fake_predict_session(tmp_path, n_workers=None, n_producers=2)
  assert session._conf.processing_conf.workers == 2
  assert any(
    "after 385 MiB for the parent and 2 producers" in r.getMessage()
    for r in caplog.records
  )


def test_an_explicit_count_is_kept_under_the_same_limit(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _simulate(monkeypatch, 1024)
  session = fake_predict_session(tmp_path, n_workers=3)
  assert session._conf.processing_conf.workers == 3


def test_the_default_is_the_core_count_when_memory_is_plentiful(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _simulate(monkeypatch, 10**6)
  session = fake_predict_session(tmp_path, n_workers=None)
  assert session._conf.processing_conf.workers == 32


def test_the_default_is_decided_after_the_other_arguments_are_checked(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  _simulate(monkeypatch, 1024)
  with caplog.at_level(logging.INFO, logger="birdnet"), pytest.raises(ValueError):
    fake_predict_session(tmp_path, n_workers=None, overlap_duration_s=3.0)
  assert not any("n_workers defaulted" in r.getMessage() for r in caplog.records)


def test_a_cgroup_limit_below_the_hosts_view_decides(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  _simulate(monkeypatch, 10**6)
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: 1024 * _MIB)
  with caplog.at_level(logging.INFO, logger="birdnet"):
    session = fake_predict_session(tmp_path, n_workers=None)
  assert session._conf.processing_conf.workers == 3
  assert any("(cgroup limit)" in r.getMessage() for r in caplog.records)


def test_a_gpu_gets_the_default_of_the_cpu(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A worker per core, fitted to the host's memory: the device's is not known."""
  _simulate(monkeypatch, 1024)
  session = fake_predict_session(tmp_path, n_workers=None, device="GPU:1")
  assert session._conf.processing_conf.workers == 3


def test_a_device_list_gets_one_worker_for_each_entry(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _simulate(monkeypatch, 10**6)
  session = fake_predict_session(
    tmp_path, n_workers=None, device=["GPU:0", "GPU:1", "CPU"]
  )
  assert session._conf.processing_conf.workers == 3


def test_an_empty_device_list_is_rejected_not_given_a_worker(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _simulate(monkeypatch, 10**6)
  with pytest.raises(ValueError, match=r"length of device list \(0\) must match"):
    fake_predict_session(tmp_path, n_workers=None, device=[])


def test_a_misspelt_device_is_rejected_before_the_default_is_decided(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  _simulate(monkeypatch, 1024)
  with caplog.at_level(logging.INFO, logger="birdnet"), pytest.raises(ValueError):
    fake_predict_session(tmp_path, n_workers=None, device="gpu")
  assert not any("n_workers defaulted" in r.getMessage() for r in caplog.records)


def _host(
  monkeypatch: pytest.MonkeyPatch, n_cores: int, available_mib: int, model_mib: int
) -> None:
  monkeypatch.setattr(sys, "platform", "linux")
  monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: n_cores)
  monkeypatch.setattr(memory, "host_memory_available", lambda: available_mib * _MIB)
  monkeypatch.setattr(memory, "cgroup_memory_available", lambda: None)
  monkeypatch.setattr(memory, "model_bytes", lambda _path: model_mib * _MIB)


def test_a_tflite_batch_no_longer_costs_workers_on_a_small_machine(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A MacBook Air M1 with 3772 MiB free started 5 of 8 workers at batch 16.

  They held 372 to 374 MiB each, where 534 MiB were planned.
  """
  from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4
  from birdnet.core.backends import TF_BACKEND_LIB_ARG

  _host(monkeypatch, n_cores=8, available_mib=3772, model_mib=49)

  n_workers = memory.default_n_workers(
    backend_type=AcousticTFBackendFP32V2_4,
    backend_kwargs={TF_BACKEND_LIB_ARG: "tflite"},
    model_path=tmp_path / "model.tflite",
    n_producers=1,
    batch_size=16,
    prefetch_ratio=1,
    segment_samples=144_000,
  )

  assert n_workers == 8


@pytest.mark.parametrize(
  ("available_mib", "device", "expected"),
  [
    # A thread per worker: a batch of 64 runs row by row in each.
    (16384, "CPU", 12),
    (8192, "CPU", 10),
    # Four fit by that estimate; with three threads each they run a batch as
    # one, and two of those fit.
    (4096, "CPU", 2),
    (8192, "GPU", 5),
  ],
)
def test_onnx_workers_are_planned_by_how_they_run_a_batch(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  available_mib: int,
  device: str,
  expected: int,
) -> None:
  from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0

  _host(monkeypatch, n_cores=12, available_mib=available_mib, model_mib=329)

  n_workers = memory.default_n_workers(
    backend_type=AcousticOnnxBackendFP32V3_0,
    backend_kwargs={},
    model_path=tmp_path / "model.onnx",
    n_producers=1,
    batch_size=64,
    prefetch_ratio=1,
    segment_samples=96_000,
    device=device,
  )

  assert n_workers == expected
