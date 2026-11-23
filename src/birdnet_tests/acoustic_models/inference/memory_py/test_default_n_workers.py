"""``n_workers=None`` fits the memory available; an explicit count is untouched."""

from __future__ import annotations

import logging
from pathlib import Path

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


def test_an_onnx_batch_with_several_rows_for_every_core_gets_one_worker(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
  from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0

  _simulate(monkeypatch, 10**6)
  kwargs = {
    "backend_kwargs": {},
    "model_path": tmp_path / "model.onnx",
    "n_producers": 1,
    "prefetch_ratio": 1,
    "segment_samples": 96_000,
  }
  with caplog.at_level(logging.INFO, logger="birdnet"):
    one = memory.default_n_workers(
      backend_type=AcousticOnnxBackendFP32V3_0, batch_size=4 * 32, **kwargs
    )
  below = memory.default_n_workers(
    backend_type=AcousticOnnxBackendFP32V3_0, batch_size=4 * 32 - 1, **kwargs
  )
  on_gpu = memory.default_n_workers(
    backend_type=AcousticOnnxBackendFP32V3_0, batch_size=4 * 32, device="GPU", **kwargs
  )

  assert (one, below, on_gpu) == (1, 32, 32)
  assert any("one onnx worker" in r.getMessage() for r in caplog.records)


def test_another_backend_keeps_a_worker_per_core_at_a_large_batch(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  _simulate(monkeypatch, 10**6)
  session = fake_predict_session(tmp_path, n_workers=None, batch_size=64)
  assert session._conf.processing_conf.workers == 32
