"""Each CPU worker of a thread-pooled backend gets its share of the cores."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from birdnet.acoustic.inference.resources import WorkerResources
from birdnet.acoustic.models.v2_4.tf import AcousticTFBackendFP32V2_4
from birdnet.acoustic.models.v3_0.onnx import AcousticOnnxBackendFP32V3_0
from birdnet.acoustic.models.v3_0.pb import AcousticPBBackendFP32V3_0
from birdnet.acoustic.models.v3_0.pt import AcousticPTBackendFP32V3_0
from birdnet.core.backends import CPU_THREADS_ARG, TF_BACKEND_LIB_ARG

pytestmark = pytest.mark.no_tf


def _backend_kwargs(
  backend_type: type,
  device: str | list[str],
  n_workers: int,
  backend_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
  config = SimpleNamespace(
    start_method="spawn",
    processing_conf=SimpleNamespace(workers=n_workers, device=device),
    model_conf=SimpleNamespace(
      path=Path("model"),
      backend_type=backend_type,
      backend_kwargs=backend_kwargs or {},
    ),
  )
  res = WorkerResources.create(config)  # type: ignore[arg-type]
  return res.backend_loader._backend_kwargs


@pytest.fixture(autouse=True)
def _eight_cores(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 8)
  monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
  monkeypatch.delenv("TF_NUM_INTRAOP_THREADS", raising=False)


@pytest.mark.parametrize(
  "backend_type",
  [AcousticOnnxBackendFP32V3_0, AcousticPTBackendFP32V3_0, AcousticPBBackendFP32V3_0],
)
@pytest.mark.parametrize(("n_workers", "expected"), [(8, 1), (4, 2), (1, 8)])
def test_cpu_workers_split_the_cores(
  backend_type: type, n_workers: int, expected: int
) -> None:
  assert _backend_kwargs(backend_type, "CPU", n_workers)[CPU_THREADS_ARG] == expected


def test_only_cpu_workers_count() -> None:
  kwargs = _backend_kwargs(AcousticOnnxBackendFP32V3_0, ["GPU:0", "CPU", "CPU"], 3)

  assert kwargs[CPU_THREADS_ARG] == 4


def test_gpu_only_session_sets_no_thread_count() -> None:
  assert CPU_THREADS_ARG not in _backend_kwargs(AcousticOnnxBackendFP32V3_0, "GPU", 2)


def test_tflite_keeps_its_own_setting() -> None:
  kwargs = _backend_kwargs(
    AcousticTFBackendFP32V2_4, "CPU", 8, {TF_BACKEND_LIB_ARG: "tflite"}
  )

  assert kwargs == {TF_BACKEND_LIB_ARG: "tflite"}


@pytest.mark.parametrize(
  "backend_type", [AcousticOnnxBackendFP32V3_0, AcousticPTBackendFP32V3_0]
)
def test_explicit_omp_num_threads_wins(
  backend_type: type, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("OMP_NUM_THREADS", "3")

  assert _backend_kwargs(backend_type, "CPU", 8)[CPU_THREADS_ARG] == 3


def test_pb_reads_tf_intraop_threads_only(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("OMP_NUM_THREADS", "3")
  monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "2")

  assert _backend_kwargs(AcousticPBBackendFP32V3_0, "CPU", 8)[CPU_THREADS_ARG] == 2


@pytest.mark.parametrize("value", ["abc", "0", "-1", ""])
def test_invalid_env_value_falls_back_to_the_share(
  value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setenv("OMP_NUM_THREADS", value)

  assert _backend_kwargs(AcousticOnnxBackendFP32V3_0, "CPU", 4)[CPU_THREADS_ARG] == 2
