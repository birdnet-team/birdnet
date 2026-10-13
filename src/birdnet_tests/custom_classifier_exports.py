"""Builds the classifier exports BirdNET-Analyzer produces, for the tests.

An export wraps the official SavedModel the way the analyzer's
``save_linear_classifier``, ``save_raven_model`` and ``save_detached_classifier``
do: a Keras model that calls the base model's signature for its embeddings (and,
in append mode, its scores), a fresh linear head and, on the 3.0 base, a sigmoid,
converted with ``TFLiteConverter.from_keras_model`` or saved with a ``basic``
signature. The head is untrained and seeded, so two exports of one spec carry the
same weights; no test depends on what the head predicts.

The build imports TensorFlow, so it runs in a subprocess (``python -m`` on this
module) and the test process stays TensorFlow-free, which the litert lane
requires. The files are cached under the app data directory next to the official
models (or under ``BIRDNET_TEST_EXPORTS_DIR``), keyed by the official model's
content and this module's source, so one build serves every pytest run and
xdist worker on the machine, including the environments without TensorFlow.

BirdNET-Analyzer's own test classifier is not built but downloaded, see
`ensure_analyzer_fixture`.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from birdnet.acoustic.models.v2_4.model import AcousticModelV2_4
from birdnet.acoustic.models.v2_4.pb import AcousticPBDownloaderV2_4
from birdnet.acoustic.models.v3_0.model import AcousticModelV3_0
from birdnet.acoustic.models.v3_0.pb import AcousticPBDownloaderV3_0
from birdnet.core.backends import tf_installed
from birdnet.utils.helper import directory_lock, get_species_from_file
from birdnet.utils.label_manifest import LabelInput, ensure_artifact
from birdnet.utils.local_data import APP_DIR, get_lang_dir, get_model_path

LABELS = ["Sci a_Common a", "Sci b_Common b", "Sci c_Common c"]

# Building an export plus the official model load it needs; the 3.0 append tflite
# alone is ~540 MB.
BUILD_TIMEOUT_S = 1500

# The exports are ~1.2 GB and rebuilt within minutes. CI points this outside
# BIRDNET_APP_DATA, the directory its model cache stores.
ENV_VAR_EXPORTS_DIR = "BIRDNET_TEST_EXPORTS_DIR"


def _cache_root() -> Path:
  override = os.getenv(ENV_VAR_EXPORTS_DIR)
  if override:
    return Path(override).expanduser().resolve()
  return APP_DIR / "test-fixtures" / "custom-classifiers"


_CACHE_ROOT = _cache_root()
# One lock for both bases, so their builds never run side by side: a 3.0 build
# peaks at ~4.7 GB RSS, a 2.4 build at ~1.3 GB, and the macOS runners have 7 GB.
_BUILD_LOCK = _CACHE_ROOT / "build.lock"

# BirdNET-Analyzer's test classifier and its labels, pinned to a commit of the
# repository the analyzer's test suite reads them from.
_ANALYZER_COMMIT = "0545aad4395b6a4ddff49f7b52cd0e34feb94032"
_ANALYZER_URL = (
  "https://raw.githubusercontent.com/birdnet-team/birdnet-test-data/"
  f"{_ANALYZER_COMMIT}/analyze/"
)
_ANALYZER_DIR = APP_DIR / "test-fixtures" / "analyzer" / _ANALYZER_COMMIT[:12]
_ANALYZER_MODEL = LabelInput(
  path=_ANALYZER_DIR / "CustomClassifier.tflite",
  url=_ANALYZER_URL + "CustomClassifier.tflite",
  size=26045016,
  sha256="f9d27f367d3d559ce338a306f8556e59efcf042edcc45e91e68337d9ad5858ca",
)
_ANALYZER_LABELS = LabelInput(
  path=_ANALYZER_DIR / "CustomClassifier_Labels.txt",
  url=_ANALYZER_URL + "CustomClassifier_Labels.txt",
  size=69,
  sha256="30f08fc0a98e838deb1b18b679dc345481621fe79594406d30b62737e8484e22",
)
ANALYZER_FIXTURE_LABELS = [
  "Cardinalis cardinalis_Northern Cardinal",
  "Cyanocitta cristata_Blue Jay",
]


@dataclass(frozen=True)
class ExportSpec:
  version: Literal["2.4", "3.0"]
  mode: Literal["replace", "append"] = "replace"
  hidden_units: int = 0
  kind: Literal["tflite", "raven", "detached"] = "tflite"

  @property
  def name(self) -> str:
    return f"cc_{self.version}_{self.mode}_{self.hidden_units}_{self.kind}"

  def encode(self) -> str:
    return f"{self.version}:{self.mode}:{self.hidden_units}:{self.kind}"

  @classmethod
  def decode(cls, text: str) -> ExportSpec:
    version, mode, hidden_units, kind = text.split(":")
    return cls(version, mode, int(hidden_units), kind)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Export:
  spec: ExportSpec
  # The .tflite file, or the SavedModel directory of a Raven export.
  model: Path
  labels: Path
  n_species: int


@dataclass(frozen=True)
class _Base:
  segment_samples: int
  embeddings_dim: int
  outputs_probabilities: bool


_BASES: dict[str, _Base] = {
  "2.4": _Base(
    AcousticModelV2_4.get_segment_size_samples(),
    AcousticModelV2_4.get_embeddings_dim(),
    outputs_probabilities=False,
  ),
  "3.0": _Base(
    AcousticModelV3_0.get_segment_size_samples(),
    AcousticModelV3_0.get_embeddings_dim(),
    outputs_probabilities=True,
  ),
}


_PB_DOWNLOADERS = {"2.4": AcousticPBDownloaderV2_4, "3.0": AcousticPBDownloaderV3_0}


def _official_pb(version: str) -> tuple[Path, Path]:
  """(SavedModel directory, English labels file) of the base; downloads if absent."""
  _PB_DOWNLOADERS[version].get_model_path_and_labels("en_us")
  return (
    get_model_path("acoustic", version, "pb", "fp32"),  # type: ignore[arg-type]
    get_lang_dir("acoustic", version, "pb") / "en_us.txt",  # type: ignore[arg-type]
  )


def _official_pb_on_disk_or_skip(version: str) -> Path:
  """The base's SavedModel directory, which names the cache; never downloads."""
  if not _PB_DOWNLOADERS[version]._check_acoustic_model_available():
    pytest.skip(
      "the classifier exports are built with TensorFlow from the official "
      f"{version} SavedModel, and neither is here"
    )
  return get_model_path("acoustic", version, "pb", "fp32")  # type: ignore[arg-type]


def _sha256(paths: Iterable[Path]) -> str:
  digest = hashlib.sha256()
  for path in paths:
    with path.open("rb") as f:
      for chunk in iter(lambda: f.read(1 << 20), b""):
        digest.update(chunk)
  return digest.hexdigest()


def _recipe_source() -> bytes:
  parts = [repr(LABELS)] + [
    inspect.getsource(part)
    for part in (
      ExportSpec,
      _signature,
      _head,
      _save_tflite,
      _save_raven,
      _save_detached,
      _write_raven_labels,
      _build,
    )
  ]
  return "\n".join(parts).encode("utf-8")


def _cache_dir(version: str, pb_dir: Path) -> Path:
  # The official model's content and the recipe decide what gets built, so
  # together they name the cache; a model or recipe change builds afresh.
  digest = hashlib.sha256(_recipe_source())
  digest.update(
    _sha256(
      [
        pb_dir / "saved_model.pb",
        pb_dir / "variables" / "variables.index",
        pb_dir / "variables" / "variables.data-00000-of-00001",
      ]
    ).encode("ascii")
  )
  return _CACHE_ROOT / f"v{version}-{digest.hexdigest()[:16]}"


def _remove_stale_caches(version: str, current: Path) -> None:
  """Drop this base's caches from other models or recipes; they are ~1 GB each."""
  for path in _CACHE_ROOT.glob(f"v{version}-*"):
    if path != current and path.is_dir():
      shutil.rmtree(path, ignore_errors=True)


def _export(spec: ExportSpec, spec_dir: Path) -> Export:
  model = spec_dir / "model" if spec.kind == "raven" else spec_dir / "model.tflite"
  labels = spec_dir / "labels.txt"
  return Export(spec, model, labels, len(get_species_from_file(labels)))


def _can_build() -> bool:
  """Whether TensorFlow is here to build with; skips where it is too old."""
  if not tf_installed():
    return False
  version = importlib.metadata.version("tensorflow")
  major, minor = (int(part) for part in version.split(".")[:2])
  if (major, minor) < (2, 18):
    pytest.skip(f"building the classifier exports needs TensorFlow 2.18, not {version}")
  return True


def ensure_exports(specs: Sequence[ExportSpec]) -> dict[ExportSpec, Export]:
  """Return the exports, building the ones that are not cached yet.

  Only the build needs TensorFlow. Without it the exports a TensorFlow
  environment left in the cache are used, and the test is skipped if any is
  missing.
  """
  can_build = _can_build()
  exports: dict[ExportSpec, Export] = {}
  for version in sorted({spec.version for spec in specs}):
    if can_build:
      pb_dir, _ = _official_pb(version)
    else:
      pb_dir = _official_pb_on_disk_or_skip(version)
    cache_dir = _cache_dir(version, pb_dir)
    wanted = [spec for spec in specs if spec.version == version]
    # A finished export is renamed into place, so it can be looked for without
    # the lock, which is for the build only.
    if _missing(cache_dir, wanted) and can_build:
      with directory_lock(
        _BUILD_LOCK, "custom classifier export build", timeout_s=BUILD_TIMEOUT_S
      ):
        missing = _missing(cache_dir, wanted)
        if missing:
          _build_in_subprocess(cache_dir, missing)
          _remove_stale_caches(version, cache_dir)
    elif _missing(cache_dir, wanted):
      pytest.skip(
        f"the {version} classifier exports are not built yet, and building them "
        "needs TensorFlow"
      )
    for spec in wanted:
      exports[spec] = _export(spec, cache_dir / spec.name)
  return exports


def _missing(cache_dir: Path, specs: Sequence[ExportSpec]) -> list[ExportSpec]:
  return [spec for spec in specs if not (cache_dir / spec.name).is_dir()]


def ensure_analyzer_fixture() -> tuple[Path, Path]:
  """(model, labels) of BirdNET-Analyzer's test classifier; downloads if absent."""
  with directory_lock(
    _ANALYZER_DIR.with_name(_ANALYZER_DIR.name + ".lock"),
    "the download of BirdNET-Analyzer's test classifier",
  ):
    for artifact in (_ANALYZER_MODEL, _ANALYZER_LABELS):
      ensure_artifact(artifact, "Downloading BirdNET-Analyzer's test classifier")
  return _ANALYZER_MODEL.path, _ANALYZER_LABELS.path


def _build_in_subprocess(cache_dir: Path, specs: Sequence[ExportSpec]) -> None:
  cache_dir.mkdir(parents=True, exist_ok=True)
  env = dict(os.environ)
  env["PYTHONPATH"] = os.pathsep.join(
    p for p in (str(Path(__file__).parents[1]), env.get("PYTHONPATH")) if p
  )
  env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
  result = subprocess.run(
    [sys.executable, "-m", __name__, str(cache_dir), *(s.encode() for s in specs)],
    env=env,
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    timeout=BUILD_TIMEOUT_S,
  )
  if result.returncode != 0:
    raise RuntimeError(
      f"Building {', '.join(s.name for s in specs)} failed (exit "
      f"{result.returncode}):\n{result.stderr[-4000:]}"
    )


# --- the build itself: runs in the subprocess only ---------------------------


def _signature(saved_model: object, version: str, output: str) -> tuple:
  """(callable, input key, output key) that yields ``output`` of the base model."""
  signatures = saved_model.signatures  # type: ignore[attr-defined]
  if version == "2.4":
    if output == "embeddings":
      return signatures["embeddings"], "inputs", "embeddings"
    return signatures["basic"], "inputs", "scores"
  return signatures["serving_default"], "x", output


def _head(keras: object, input_size: int, n_labels: int, hidden_units: int) -> object:
  """The classifier as the analyzer hands it to its export: no activation."""
  import keras as keras_lib

  regularizer = keras_lib.regularizers.l2(1e-5)
  model = keras_lib.Sequential()
  model.add(keras_lib.layers.InputLayer(shape=(input_size,)))
  if hidden_units > 0:
    model.add(
      keras_lib.layers.Dense(
        hidden_units,
        activation="relu",
        kernel_regularizer=regularizer,
        kernel_initializer="he_normal",
      )
    )
  model.add(
    keras_lib.layers.Dense(
      n_labels, kernel_regularizer=regularizer, kernel_initializer="glorot_uniform"
    )
  )
  return model


def _save_tflite(
  saved_model: object, head: object, spec: ExportSpec, base: _Base, path: Path
) -> None:
  import keras
  import tensorflow as tf

  class SignatureLayer(keras.layers.Layer):  # type: ignore[misc,name-defined]
    def __init__(self, output: str) -> None:
      super().__init__()
      self.fn, self.input_key, self.output_key = _signature(
        saved_model, spec.version, output
      )

    def call(self, inputs: object) -> object:
      return self.fn(**{self.input_key: inputs})[self.output_key]

  if base.outputs_probabilities:
    inputs = keras.Input(shape=(base.segment_samples,), dtype=tf.float32, name="x")
    embeddings = SignatureLayer("embeddings")(inputs)
    output = keras.layers.Activation("sigmoid")(head(embeddings))  # type: ignore[operator]
    if spec.mode == "append":
      output = keras.layers.concatenate([SignatureLayer("predictions")(inputs), output])
    outputs = {
      "predictions": keras.layers.Identity(name="predictions")(output),
      "embeddings": keras.layers.Identity(name="embeddings")(embeddings),
    }
  else:
    inputs = keras.Input(
      shape=(base.segment_samples,), dtype=tf.float32, name="input_audio"
    )
    embeddings = SignatureLayer("embeddings")(inputs)
    outputs = head(embeddings)  # type: ignore[operator]
    if spec.mode == "append":
      outputs = keras.layers.concatenate(
        [SignatureLayer("predictions")(inputs), outputs], name="combined_output"
      )

  combined = keras.Model(inputs=inputs, outputs=outputs, name="basic")
  converter = tf.lite.TFLiteConverter.from_keras_model(combined)
  path.write_bytes(converter.convert())


def _save_raven(
  saved_model: object, head: object, spec: ExportSpec, base: _Base, path: Path
) -> None:
  import keras
  import tensorflow as tf

  class Combined(keras.Model):  # type: ignore[misc,name-defined]
    # Both parts are attributes so that saving tracks their variables.
    def __init__(self) -> None:
      super().__init__(name=f"combined_model_{spec.mode}_sigmoid")
      self.pb_model = saved_model
      self.classifier = head

    def call(self, inputs: object) -> object:
      fn, in_key, out_key = _signature(self.pb_model, spec.version, "embeddings")
      classified = tf.sigmoid(self.classifier(fn(**{in_key: inputs})[out_key]))
      if spec.mode != "append":
        return classified
      fn, in_key, out_key = _signature(self.pb_model, spec.version, "predictions")
      scores = fn(**{in_key: inputs})[out_key]
      if not base.outputs_probabilities:
        scores = tf.sigmoid(scores)
      return tf.concat([scores, classified], axis=-1)

  combined = Combined()

  @tf.function(
    input_signature=[
      tf.TensorSpec(shape=[None, base.segment_samples], dtype=tf.float32)
    ]
  )
  def basic(inputs: object) -> dict[str, object]:
    return {"scores": combined(inputs)}

  tf.saved_model.save(combined, str(path), signatures={"basic": basic})
  # Resaved with the inference signatures only, as the analyzer does for Raven.
  loaded = tf.saved_model.load(str(path))
  inference_sigs = {
    k: loaded.signatures[k] for k in loaded.signatures if k != "__saved_model_init_op"
  }
  tf.saved_model.save(loaded, str(path), signatures=inference_sigs)


def _save_detached(head: object, base: _Base, path: Path) -> None:
  import keras
  import tensorflow as tf

  inputs = keras.Input(
    shape=(base.embeddings_dim,), dtype=tf.float32, name="detached_input"
  )
  outputs = head(inputs)  # type: ignore[operator]
  if base.outputs_probabilities:
    outputs = keras.layers.Activation("sigmoid")(outputs)
  detached = keras.Model(inputs=inputs, outputs=outputs, name="detached_classifier")
  converter = tf.lite.TFLiteConverter.from_keras_model(detached)
  path.write_bytes(converter.convert())


def _write_raven_labels(model_dir: Path, labels: Sequence[str]) -> None:
  import csv

  label_ids = [label[:4].replace(" ", "") + str(i) for i, label in enumerate(labels, 1)]
  (model_dir / "labels").mkdir(exist_ok=True)
  with (model_dir / "labels" / "label_names.csv").open("w", newline="") as f:
    csv.writer(f).writerows(zip(label_ids, labels, strict=True))


def _build(cache_dir: Path, specs: Sequence[ExportSpec]) -> None:
  import keras
  import tensorflow as tf

  for version in sorted({spec.version for spec in specs}):
    base = _BASES[version]
    pb_dir, labels_file = _official_pb(version)
    saved_model = tf.saved_model.load(str(pb_dir))
    official_labels = list(get_species_from_file(labels_file))
    for spec in (s for s in specs if s.version == version):
      keras.utils.set_random_seed(0)
      head = _head(keras, base.embeddings_dim, len(LABELS), spec.hidden_units)
      building = cache_dir / (spec.name + ".building")
      shutil.rmtree(building, ignore_errors=True)
      building.mkdir(parents=True)
      labels = LABELS if spec.mode == "replace" else [*official_labels, *LABELS]
      if spec.kind == "tflite":
        _save_tflite(saved_model, head, spec, base, building / "model.tflite")
      elif spec.kind == "raven":
        _save_raven(saved_model, head, spec, base, building / "model")
        _write_raven_labels(building / "model", labels)
      else:
        _save_detached(head, base, building / "model.tflite")
        labels = LABELS
      (building / "labels.txt").write_text(
        "".join(label + "\n" for label in labels), encoding="utf-8"
      )
      os.replace(building, cache_dir / spec.name)


if __name__ == "__main__":
  _build(Path(sys.argv[1]), [ExportSpec.decode(arg) for arg in sys.argv[2:]])
