"""Which model `birdnet-benchmark` loads, and that it runs the one it loaded.

The models are stand-ins: what is pinned is the dispatch from the command's
options to the loader and to `predict`.
"""

from argparse import Namespace
from typing import Any

import pytest

from birdnet_benchmark import cli

pytestmark = pytest.mark.no_tf


class _Model:
  def __init__(self) -> None:
    self.predict_kwargs: dict[str, Any] | None = None

  def get_sig_fmin(self) -> int:
    return 0

  def get_sig_fmax(self) -> int:
    return 15_000

  def predict(self, inputs: object, **kwargs: Any) -> str:  # noqa: ANN401
    self.predict_kwargs = kwargs
    return "output"


def _options(**changed: object) -> Namespace:
  options = {
    "inputs": {"audio.wav"},
    "model_version": "2.4",
    "backend": "tf",
    "tf_library": "tflite",
    "precision": "fp32",
    "producers": 1,
    "workers": 2,
    "top_k": 5,
    "batch_size": 1,
    "half_precision": False,
    "devices": ["CPU"],
    "overlap": 0.0,
    "confidence": 0.1,
    "prefetch_ratio": 1,
    "speed": 1.0,
    "show_stats": "benchmark",
    "use_perch": False,
  }
  return Namespace(**{**options, **changed})


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[cli.BenchmarkResultContainer]:
  containers: list[cli.BenchmarkResultContainer] = []
  monkeypatch.setattr(cli, "save_statistics", containers.append)
  return containers


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict, _Model]]:
  """Every call of the two loaders, with the stand-in each one returned."""
  calls: list[tuple[tuple, dict, _Model]] = []

  def load(*args: object, **kwargs: object) -> _Model:
    calls.append((args, kwargs, _Model()))
    return calls[-1][2]

  monkeypatch.setattr(cli.birdnet.model_loader, "load", load)
  monkeypatch.setattr(cli, "load_perch_v2", load)
  return calls


def test_perch_is_loaded_and_run(
  loaded: list[tuple[tuple, dict, _Model]], saved: list[cli.BenchmarkResultContainer]
) -> None:
  cli.run_benchmark_from_ns(_options(use_perch=True, backend="pb", devices=["GPU"]))

  [(args, kwargs, model)] = loaded
  assert (args, kwargs) == ((), {"device": "GPU"})
  assert model.predict_kwargs is not None
  assert model.predict_kwargs["apply_sigmoid"] is False
  assert model.predict_kwargs["sigmoid_sensitivity"] is None
  assert [container.output for container in saved] == ["output"]


def test_the_3_0_model_runs_on_the_onnx_backend(
  loaded: list[tuple[tuple, dict, _Model]], saved: list[cli.BenchmarkResultContainer]
) -> None:
  cli.run_benchmark_from_ns(_options(model_version="3.0", backend="onnx"))

  [(args, kwargs, model)] = loaded
  assert args == ("acoustic", "3.0", "onnx")
  assert kwargs == {"precision": "fp32"}
  assert model.predict_kwargs is not None
  assert model.predict_kwargs["apply_sigmoid"] is True
  assert len(saved) == 1


def test_the_tf_backend_gets_its_library(
  loaded: list[tuple[tuple, dict, _Model]], saved: list[cli.BenchmarkResultContainer]
) -> None:
  cli.run_benchmark_from_ns(_options(tf_library="litert", precision="fp16"))

  [(args, kwargs, _)] = loaded
  assert args == ("acoustic", "2.4", "tf")
  assert kwargs == {"precision": "fp16", "library": "litert"}
