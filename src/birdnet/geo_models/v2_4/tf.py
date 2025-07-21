from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, final

import numpy as np

from birdnet.geo_models.base import (
  GeoInferenceBackend,
)
from birdnet.helper import load_litert_model, load_tf_model
from birdnet.translations import AVAILABLE_LANGUAGES_V2_4

if TYPE_CHECKING:
  from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
  from tensorflow.lite.python.interpreter import Interpreter as TFInterpreter


import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Literal, final

from ordered_set import OrderedSet

from birdnet.base import (
  LIBRARY_LITERT,
  LIBRARY_TF,
  LIBRARY_TYPES,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_PRECISIONS,
)
from birdnet.geo_models.base import GeoInferenceBackend
from birdnet.geo_models.v2_4.base import GeoModelBaseV2_4
from birdnet.helper import ModelInfo, load_litert_model
from birdnet.local_data import get_local_model_root_dir
from birdnet.utils import download_file_tqdm, get_species_from_file

# All meta models are same for all precisions and int8 is the smallest download
model_info = ModelInfo(
  dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_int8.zip",
  dl_file_name="meta-model.tflite",
  dl_size=45948867,
  file_size=29526096,
)


class GeoTFDownloaderV2_4:
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_root = get_local_model_root_dir(
      GeoTFModelV2_4.get_model_type(),
      GeoTFModelV2_4.get_version(),
      GeoTFModelV2_4.get_backend(),
    )

    model_path = model_root / "model.tflite"
    lang_dir = model_root / "labels"
    return model_path, lang_dir

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()

    if not model_path.is_file():
      return False

    file_stats = os.stat(model_path)
    is_newest_version = file_stats.st_size == model_info.file_size
    if not is_newest_version:
      return False

    if not lang_dir.is_dir():
      return False

    return all(
      (lang_dir / f"{lang}.txt").is_file() for lang in AVAILABLE_LANGUAGES_V2_4
    )

  @classmethod
  def _download_geo_model(cls) -> None:
    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        model_info.dl_url,
        zip_download_path,
        download_size=model_info.dl_size,
        description="Downloading model",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_path = extract_dir / model_info.dl_file_name
      species_dl_dir = extract_dir / "labels"

      geo_model_path, geo_lang_dir = cls._get_paths()
      geo_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(geo_model_dl_path, geo_model_path)

      geo_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(geo_lang_dir, ignore_errors=True)
      shutil.move(species_dl_dir, geo_lang_dir)

  @classmethod
  def get_model_path_and_labels(cls, lang_id: str) -> tuple[Path, OrderedSet[str]]:
    assert lang_id in AVAILABLE_LANGUAGES_V2_4
    if not cls._check_geo_model_available():
      cls._download_geo_model()
    assert cls._check_geo_model_available()

    model_path, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang_id}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang_id}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class GeoTFModelV2_4(GeoModelBaseV2_4):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
  ) -> None:
    super().__init__(model_path, species_list)

  @final
  @classmethod
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_TF

  @classmethod
  def load_official(
    cls,
    lang_id: MODEL_LANGUAGES,
  ) -> GeoTFModelV2_4:
    model_path, species_list = GeoTFDownloaderV2_4.get_model_path_and_labels(lang_id)
    result = GeoTFModelV2_4(model_path, species_list)
    return result


class TFGeoInferenceBackend(GeoInferenceBackend):
  def __init__(self, model_path: Path, inference_library: LIBRARY_TYPES) -> None:
    super().__init__()
    self._model_path = model_path
    self._interp: TFLiteInterpreter | TFInterpreter | None = None
    self._inference_library = inference_library
    self._in_idx: int | None = None
    self._out_idx: int | None = None
    self._cached_shape: tuple[int, ...] | None = None

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return True

  def load(self) -> None:
    assert self._interp is None
    if self._inference_library == LIBRARY_TF:
      self._interp = load_tf_model(self._model_path, allocate_tensors=True)
    elif self._inference_library == LIBRARY_LITERT:
      self._interp = load_litert_model(self._model_path, allocate_tensors=True)
    else:
      raise AssertionError()

    self._in_idx = self._interp.get_input_details()[0]["index"]  # type: ignore
    self._out_idx = self._interp.get_output_details()[0]["index"]  # type: ignore

  def _set_tensor(self, batch: np.ndarray) -> None:
    assert self._interp is not None
    assert batch.flags["C_CONTIGUOUS"]
    assert batch.ndim == 2
    assert self._interp is not None

    shape = batch.shape
    if self._cached_shape != shape:
      self._interp.resize_tensor_input(self._in_idx, shape, strict=True)
      self._interp.allocate_tensors()
      self._cached_shape = shape
    # self._in_view[:n, :] = batch
    self._interp.set_tensor(self._in_idx, batch)

  @final
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray:
    # TODO: implement load on different CPUs
    if "CPU" not in device_name:
      raise ValueError("TensorFlow models can only be loaded on CPU!")

    assert self._interp is not None
    self._set_tensor(batch)
    self._interp.invoke()
    res: np.ndarray = self._interp.get_tensor(self._out_idx)
    assert res.dtype == np.float32
    return res
