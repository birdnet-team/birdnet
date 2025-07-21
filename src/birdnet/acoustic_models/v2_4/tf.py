from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, final

import numpy as np

from birdnet.acoustic_models.base import (
  AcousticInferenceBackend,
  AcousticInferenceBackendLoader,
)
from birdnet.acoustic_models.inference.prediction_result import PredictionResult
from birdnet.helper import load_litert_model, load_tf_model
from birdnet.io_lock import IOLockHandler

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

from birdnet.acoustic_models.base import AcousticInferenceBackend
from birdnet.acoustic_models.v2_4.base import AVAILABLE_LANGUAGES, AcousticModelBaseV2_4
from birdnet.base import (
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_PRECISION_FLOAT16,
  MODEL_PRECISION_FLOAT32,
  MODEL_PRECISION_INT8,
  MODEL_PRECISIONS,
)
from birdnet.helper import ModelInfo, load_litert_model
from birdnet.local_data import get_local_model_root_dir
from birdnet.utils import download_file_tqdm, get_species_from_file

models = {
  MODEL_PRECISION_INT8: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_int8.zip",
    dl_file_name="audio-model-int8.tflite",
    dl_size=45948867,
    file_size=41064296,
  ),
  MODEL_PRECISION_FLOAT16: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite_fp16.zip",
    dl_file_name="audio-model-fp16.tflite",
    dl_size=53025528,
    file_size=25932528,
  ),
  MODEL_PRECISION_FLOAT32: ModelInfo(
    dl_url="https://zenodo.org/records/15050749/files/BirdNET_v2.4_tflite.zip",
    dl_file_name="audio-model.tflite",
    dl_size=76822925,
    file_size=51726412,
  ),
}


class AcousticTFDownloaderV2_4:
  @classmethod
  def _get_paths(cls, precision: MODEL_PRECISIONS) -> tuple[Path, Path]:
    model_root = get_local_model_root_dir(
      AcousticTFModelV2_4.get_model_type(),
      AcousticTFModelV2_4.get_version(),
      AcousticTFModelV2_4.get_backend(),
    )

    model_path = model_root / f"model-{precision}.tflite"
    lang_dir = model_root / "labels"
    return model_path, lang_dir

  @classmethod
  def _check_acoustic_model_available(cls, precision: MODEL_PRECISIONS) -> bool:
    model_path, lang_dir = cls._get_paths(precision)

    if not model_path.is_file():
      return False

    file_stats = os.stat(model_path)
    is_newest_version = file_stats.st_size == models[precision].file_size
    if not is_newest_version:
      return False

    if not lang_dir.is_dir():
      return False

    return all((lang_dir / f"{lang}.txt").is_file() for lang in AVAILABLE_LANGUAGES)

  @classmethod
  def _download_acoustic_model(cls, precision: MODEL_PRECISIONS) -> None:
    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        models[precision].dl_url,
        zip_download_path,
        download_size=models[precision].dl_size,
        description="Downloading model",
      )

      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      acoustic_model_dl_path = extract_dir / models[precision].dl_file_name
      species_dl_dir = extract_dir / "labels"

      acoustic_model_path, acoustic_lang_dir = cls._get_paths(precision)
      acoustic_model_path.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(acoustic_model_dl_path, acoustic_model_path)

      acoustic_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.rmtree(acoustic_lang_dir, ignore_errors=True)
      shutil.move(species_dl_dir, acoustic_lang_dir)

  @classmethod
  def get_model_path_and_labels(
    cls, lang_id: str, precision: MODEL_PRECISIONS
  ) -> tuple[Path, OrderedSet[str]]:
    assert lang_id in AVAILABLE_LANGUAGES
    if not cls._check_acoustic_model_available(precision):
      cls._download_acoustic_model(precision)
    assert cls._check_acoustic_model_available(precision)

    model_path, langs_path = cls._get_paths(precision)

    lang_file = langs_path / f"{lang_id}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang_id}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_path, labels


class AcousticTFModelV2_4(AcousticModelBaseV2_4):
  def __init__(self) -> None:
    super().__init__()

  @final
  @classmethod
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_TF

  @final
  @classmethod
  def get_inference_backend_type(cls) -> type[AcousticInferenceBackend]:
    return AcousticTFBackend

  @final
  def get_inference_backend_args(self) -> dict:
    return {
      "model_path": self.model_path,
      "inference_library": None,
    }

  @classmethod
  def load_official(
    cls,
    lang_id: MODEL_LANGUAGES,
    precision: MODEL_PRECISIONS,
  ) -> AcousticTFModelV2_4:
    result = AcousticTFModelV2_4()
    result._model_path, result._species_list = (
      AcousticTFDownloaderV2_4.get_model_path_and_labels(lang_id, precision)
    )
    result._use_custom_model = False
    result._precision = precision
    return result

  @classmethod
  def load_custom(
    cls, model_path: Path, species_list: Path, precision: MODEL_PRECISIONS
  ) -> AcousticTFModelV2_4:
    assert model_path.is_file()
    assert species_list.is_file()

    interp = load_litert_model(model_path, allocate_tensors=False)

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    n_species_in_model = interp.get_output_details()[0]["shape"][1]
    if n_species_in_model != len(loaded_species_list):
      raise ValueError(
        f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
      )

    result = AcousticTFModelV2_4()
    result._model_path = model_path
    result._species_list = loaded_species_list
    result._use_custom_model = True
    result._precision = precision
    return result

  def analyze(
    self,
    inp: Path | str | Iterable[Path | str],
    /,
    *,
    top_k: int | None = 5,
    feeders: int = 1,
    workers: int = 4,
    batch_size: int = 1,
    prefetch_ratio: int = 1,
    overlap_duration_s: float = 0,
    default_confidence_threshold: float | None = 0.1,
    custom_confidence_thresholds: dict[str, float] | None = None,
    use_bandpass: bool = False,
    bandpass_fmin: int | None = None,
    bandpass_fmax: int | None = None,
    apply_sigmoid: bool = True,
    sigmoid_sensitivity: float | None = 1.0,
    custom_species_list: set[str] | None = None,
    half_precision: bool = True,
    max_audio_duration_min: float | None = None,
    show_stats: Literal["no", "minimal", "progress", "benchmark"] = "no",
    serial_io: bool = False,
    inference_library: Literal["tf", "litert"] = "tf",
  ) -> PredictionResult:
    # backend_loader = AcousticInferenceBackendLoader(
    #   backend_kwargs={
    #     "model_path": self.model_path,
    #     "inference_library": inference_library,
    #   }
    #   backend_type= AcousticTFBackend,
    # )
    return super().analyze(
      inp,
      top_k=top_k,
      feeders=feeders,
      workers=workers,
      batch_size=batch_size,
      prefetch_ratio=prefetch_ratio,
      overlap_duration_s=overlap_duration_s,
      default_confidence_threshold=default_confidence_threshold,
      custom_confidence_thresholds=custom_confidence_thresholds,
      use_bandpass=use_bandpass,
      bandpass_fmin=bandpass_fmin,
      bandpass_fmax=bandpass_fmax,
      apply_sigmoid=apply_sigmoid,
      sigmoid_sensitivity=sigmoid_sensitivity,
      custom_species_list=custom_species_list,
      half_precision=half_precision,
      max_audio_duration_min=max_audio_duration_min,
      show_stats=show_stats,
      device="CPU",
      serial_io=serial_io,
      inference_library=inference_library,
    )


class AcousticTFBackend(AcousticInferenceBackend):
  def __init__(
    self, model_path: Path, inference_library: Literal["tf", "litert"]
  ) -> None:
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
    if self._inference_library == "tf":
      self._interp = load_tf_model(self._model_path, allocate_tensors=True)
    elif self._inference_library == "litert":
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
