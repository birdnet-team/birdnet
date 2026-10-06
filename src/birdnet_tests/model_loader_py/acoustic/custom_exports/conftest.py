"""Real BirdNET-Analyzer exports for the custom classifier tests.

Built once per machine by `birdnet_tests.custom_classifier_exports` (see its
docstring); the first test that needs a base version pays for that version's
set, so these modules carry a timeout above the build budget.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import birdnet
from birdnet.acoustic.models.base import AcousticModelBase
from birdnet_tests.custom_classifier_exports import (
  BUILD_TIMEOUT_S,
  Export,
  ExportSpec,
  ensure_exports,
)
from birdnet_tests.test_files import TEST_FILE_SHORT

V2_4_REPLACE = ExportSpec("2.4", "replace", 0)
V2_4_APPEND = ExportSpec("2.4", "append", 0)
V2_4_REPLACE_HIDDEN = ExportSpec("2.4", "replace", 128)
V2_4_DETACHED = ExportSpec("2.4", "replace", 0, "detached")
V3_0_REPLACE = ExportSpec("3.0", "replace", 0)
V3_0_APPEND = ExportSpec("3.0", "append", 0)
V3_0_RAVEN = ExportSpec("3.0", "replace", 0, "raven")
V3_0_DETACHED = ExportSpec("3.0", "replace", 0, "detached")

EXPORT_SETS: dict[str, list[ExportSpec]] = {
  "2.4": [V2_4_REPLACE, V2_4_APPEND, V2_4_REPLACE_HIDDEN, V2_4_DETACHED],
  "3.0": [V3_0_REPLACE, V3_0_APPEND, V3_0_RAVEN, V3_0_DETACHED],
}

EXPORT_TEST_TIMEOUT_S = BUILD_TIMEOUT_S + 300

# One 7.3 s file: three segments.
N_SEGMENTS = 3

GetExport = Callable[[ExportSpec], Export]


@pytest.fixture(scope="session")
def get_export() -> GetExport:
  built: dict[ExportSpec, Export] = {}

  def _get(spec: ExportSpec) -> Export:
    if spec not in built:
      built.update(ensure_exports(EXPORT_SETS[spec.version]))
    return built[spec]

  return _get


def scores_in_model_order(model: AcousticModelBase, **kwargs: object) -> np.ndarray:
  """Every score of every segment, species in model order; shape (1, segments, n)."""
  res = model.predict(
    TEST_FILE_SHORT,
    top_k=None,
    n_workers=1,
    default_confidence_threshold=-float("inf"),
    **kwargs,  # type: ignore[arg-type]
  )
  order = np.argsort(res.species_ids, axis=-1)
  return np.take_along_axis(res.species_probs, order, axis=-1)


def embeddings(model: AcousticModelBase, **kwargs: object) -> np.ndarray:
  with model.encode_session(n_workers=1, **kwargs) as session:  # type: ignore[arg-type]
    return session.run(TEST_FILE_SHORT).embeddings


def load_unchecked(
  model: Path, labels: Path, version: str, **kwargs: object
) -> AcousticModelBase:
  """`load_custom` of a TFLite export without the validity check.

  The check opens the file in a subprocess (~5 s) and the worker resolves the
  tensors itself either way, so a test that is not about the check skips it.
  """
  return birdnet.load_custom(
    "acoustic",
    version,
    "tf",
    model,
    labels,
    check_validity=False,
    **kwargs,  # type: ignore[arg-type]
  )


def load_export_unchecked(export: Export, **kwargs: object) -> AcousticModelBase:
  return load_unchecked(export.model, export.labels, export.spec.version, **kwargs)


# Each is one pipeline run (~8 s) per pytest process, however many tests compare
# against it.
@functools.cache
def official_scores(version: str, library: str = "tflite") -> np.ndarray:
  return scores_in_model_order(birdnet.load("acoustic", version, "tf", library=library))


@functools.cache
def official_embeddings(version: str, library: str = "tflite") -> np.ndarray:
  return embeddings(birdnet.load("acoustic", version, "tf", library=library))
