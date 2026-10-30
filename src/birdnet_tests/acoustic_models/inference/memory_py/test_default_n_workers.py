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
