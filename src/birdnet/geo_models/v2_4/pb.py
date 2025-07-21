# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, final

import numpy as np

# You'll need these imports in your own code
# Next two import lines for this demo only
from ordered_set import OrderedSet

from birdnet.base import (
  MODEL_BACKEND_PB,
  MODEL_BACKENDS,
)
from birdnet.geo_models.base import GeoInferenceBackend
from birdnet.geo_models.v2_4.base import GeoModelBaseV2_4
from birdnet.local_data import get_local_model_root_dir
from birdnet.logging_utils import get_logger
from birdnet.translations import AVAILABLE_LANGUAGES_V2_4
from birdnet.utils import download_file_tqdm, get_species_from_file


def check_protobuf_model_files_exist(folder: Path) -> bool:
  exists = True
  exists &= (folder / "saved_model.pb").is_file()
  exists &= (folder / "variables").is_dir()
  exists &= (folder / "variables" / "variables.data-00000-of-00001").is_file()
  exists &= (folder / "variables" / "variables.index").is_file()
  return exists


class GeoPBDownloaderV2_4:
  @classmethod
  def _get_paths(cls) -> tuple[Path, Path]:
    model_root = get_local_model_root_dir(
      GeoPBModelV2_4.get_model_type(),
      GeoPBModelV2_4.get_version(),
      GeoPBModelV2_4.get_backend(),
    )

    model_dir = model_root / "model"
    lang_dir = model_root / "labels"
    return model_dir, lang_dir

  @classmethod
  def _check_geo_model_available(cls) -> bool:
    model_path, lang_dir = cls._get_paths()

    model_is_downloaded = True
    model_is_downloaded &= model_path.is_dir()
    model_is_downloaded &= check_protobuf_model_files_exist(model_path)

    model_is_downloaded &= lang_dir.is_dir()
    for lang in AVAILABLE_LANGUAGES_V2_4:
      model_is_downloaded &= (lang_dir / f"{lang}.txt").is_file()

    return model_is_downloaded

  @classmethod
  def _download_geo_model(cls) -> None:
    dl_url = "https://zenodo.org/records/15050749/files/BirdNET_v2.4_protobuf.zip"
    dl_size = 124522908

    with tempfile.TemporaryDirectory(prefix="birdnet_download") as temp_dir:
      zip_download_path = Path(temp_dir) / "download.zip"
      download_file_tqdm(
        dl_url,
        zip_download_path,
        download_size=dl_size,
        description="Downloading model",
      )

      print("Extracting models...")
      extract_dir = Path(temp_dir) / "extracted"

      with zipfile.ZipFile(zip_download_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

      geo_model_dl_dir = extract_dir / "meta-model"
      species_dl_dir = extract_dir / "labels"

      geo_model_dir, geo_lang_dir = cls._get_paths()
      geo_model_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(geo_model_dl_dir, geo_model_dir)

      geo_lang_dir.parent.mkdir(parents=True, exist_ok=True)
      shutil.move(species_dl_dir, geo_lang_dir)
      print("Models extracted.")

  @classmethod
  def get_model_path_and_labels(
    cls,
    lang: str,
  ) -> tuple[Path, OrderedSet[str]]:
    if not cls._check_geo_model_available():
      cls._download_geo_model()
    assert cls._check_geo_model_available()

    model_dir, langs_path = cls._get_paths()

    lang_file = langs_path / f"{lang}.txt"
    if not lang_file.is_file():
      raise ValueError(f"Language does not exist: {lang}")

    labels = get_species_from_file(lang_file, encoding="utf8")
    return model_dir, labels


class GeoPBModelV2_4(GeoModelBaseV2_4):
  def __init__(self, model_path: Path, species_list: OrderedSet[str]) -> None:
    super().__init__(model_path, species_list)

  @classmethod
  @final
  def get_backend(cls) -> MODEL_BACKENDS:
    return MODEL_BACKEND_PB

  @classmethod
  def load_official(cls, lang: str) -> GeoPBModelV2_4:
    model_path, species_list = GeoPBDownloaderV2_4.get_model_path_and_labels(lang)
    result = GeoPBModelV2_4(
      model_path=model_path,
      species_list=species_list,
    )
    return result


class PBGeoInferenceBackend(GeoInferenceBackend):
  def __init__(self, model_path: Path) -> None:
    super().__init__()
    self._model_path = str(model_path.absolute())
    self._cached_logical_device: Any | None = None
    self._infer_fn: Callable | None = None
    self._cached_device_name: str | None = None

  @final
  @classmethod
  def supports_cow(cls) -> bool:
    return False

  @final
  def load(self) -> None:
    import absl.logging

    absl_verbosity_before = absl.logging.get_verbosity()
    absl.logging.set_verbosity(absl.logging.ERROR)
    tf_verbosity_before = logging.getLogger("tensorflow").level
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.random.set_seed(0)

    # Note: memory growth needs to be set before loading the model and maybe only once in the main process
    # physical_gpu_device = gpus_with_name[0]
    # if tf.config.experimental.get_memory_growth(physical_gpu_device) is False:
    #   tf.config.experimental.set_memory_growth(physical_gpu_device, True)

    start = time.perf_counter()
    audio_model = tf.saved_model.load(self._model_path)
    end = time.perf_counter()
    logger = get_logger(__name__)
    logger.debug(f"Model loaded from {self._model_path} in {end - start:.2f} seconds.")

    absl.logging.set_verbosity(absl_verbosity_before)
    logging.getLogger("tensorflow").setLevel(tf_verbosity_before)

    self._infer_fn = audio_model.signatures["basic"]  # type: ignore

  def _set_logical_device(self, device_name: str) -> None:
    assert "GPU" in device_name or "CPU" in device_name
    import tensorflow as tf

    if "GPU" in device_name:
      physical_devices = tf.config.list_physical_devices("GPU")
      if len(physical_devices) == 0:
        raise ValueError(
          "No GPU found! Please check your TensorFlow installation and ensure that a GPU is available."
        )

      gpus_with_name = [gpu for gpu in physical_devices if device_name in gpu.name]

      if len(gpus_with_name) == 0:
        raise ValueError(f"No GPU with name '{device_name}' found!")

      self._cached_logical_device = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ][0]

    elif "CPU" in device_name:
      all_devices_with_name: list = [
        log_dev
        for log_dev in tf.config.list_logical_devices()
        if device_name in log_dev.name
      ]
      if len(all_devices_with_name) == 0:
        raise ValueError(f"No CPU with name '{device_name}' found!")
      self._cached_logical_device = all_devices_with_name[0]
    else:
      raise ValueError(f"Unsupported device name: {device_name}")

  @final
  def infer(self, batch: np.ndarray, device_name: str) -> np.ndarray:
    if self._cached_device_name is None or self._cached_device_name != device_name:
      self._set_logical_device(device_name)
      self._cached_device_name = device_name

    assert self._cached_logical_device is not None
    assert self._infer_fn is not None
    from tensorflow import Tensor, device, float32

    with device(self._cached_logical_device.name):  # type: ignore
      # prediction = self._audio_model.basic(batch)["scores"]
      predictions = self._infer_fn(inputs=batch)
    scores: Tensor = predictions["scores"]
    assert scores.dtype == float32
    scores_np = scores.numpy()  # type: ignore
    assert scores_np.dtype == np.float32
    return scores_np
