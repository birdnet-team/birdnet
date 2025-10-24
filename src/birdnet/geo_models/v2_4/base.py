from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self, final

import numpy as np
from numpy.typing import DTypeLike
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference2.backends2 import (
  InferenceBackendLoader2,
  PBInferenceBackend2,
  TFInferenceBackend2,
  VersionedInferenceBackendProtocol,
  check_pb_model_can_be_loaded,
  check_tf_model_can_be_loaded,
)
from birdnet.geo_models.base import GeoModelBase, GeoModelBase2
from birdnet.geo_models.inference.prediction_result import PredictionResult
from birdnet.globals import (
  GEO_MODEL_VERSION_V2_4,
  GEO_MODEL_VERSIONS,
  LIBRARY_TYPES,
  MODEL_BACKENDS,
  MODEL_LANGUAGES,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
)
from birdnet.helper import uint_dtype_for
from birdnet.utils import get_species_from_file


class GeoDownloaderBaseV2_4:
  AVAILABLE_LANGUAGES: OrderedSet[str] = OrderedSet(
    (
      "af",
      "ar",
      "cs",
      "da",
      "de",
      "en_uk",
      "en_us",
      "es",
      "fi",
      "fr",
      "hu",
      "it",
      "ja",
      "ko",
      "nl",
      "no",
      "pl",
      "pt",
      "ro",
      "ru",
      "sk",
      "sl",
      "sv",
      "th",
      "tr",
      "uk",
      "zh",
    )
  )


class GeoModelBaseV2_4(GeoModelBase):
  def __init__(
    self, model_path: Path, species_list: OrderedSet[str], use_custom_model: bool
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)

  @classmethod
  @final
  def get_version(cls) -> GEO_MODEL_VERSIONS:
    return GEO_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO

  def _predict(
    self,
    latitude: float,
    longitude: float,
    backend_kwargs: dict,
    /,
    *,
    week: int | None = None,
    min_confidence: float = 0.03,
    device: str = "CPU",
    half_precision: bool = True,
  ) -> PredictionResult:
    if not -90 <= latitude <= 90:
      raise ValueError(
        "Value for 'latitude' is invalid! It needs to be in interval [-90, 90]."
      )

    if not -180 <= longitude <= 180:
      raise ValueError(
        "Value for 'longitude' is invalid! It needs to be in interval [-180, 180]."
      )

    if not 0 <= min_confidence < 1.0:
      raise ValueError(
        "Value for 'min_confidence' is invalid! It needs to be in interval [0.0, 1.0)."
      )

    if week is not None and not (1 <= week <= 48):
      raise ValueError(
        "Value for 'week' is invalid! It needs to be either None or in interval [1, 48]."
      )

    if week is None:
      week = -1
    assert week is not None

    sample = np.expand_dims(np.array([latitude, longitude, week], dtype=np.float32), 0)

    backend_type = self.get_backend_type()

    try:
      backend = backend_type(**backend_kwargs)
      backend.load()
    except Exception as exc:
      raise ValueError("Failed to load backend.") from exc

    prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

    res = backend.infer(sample, device_name=device)
    assert res.dtype == np.float32
    res = res.astype(prob_dtype, copy=False)

    res = np.squeeze(res, axis=0)

    species_ids = np.arange(
      len(self.species_list),
      dtype=uint_dtype_for(
        max(0, len(self.species_list) - 1),
      ),
    )

    invalid_mask = res < min_confidence
    prediction = PredictionResult(
      species_list=self.species_list,
      species_probs=res,
      species_ids=species_ids,
      species_masked=invalid_mask,
    )

    return prediction


class TFGeoInferenceBackendV2_4(TFInferenceBackend2, VersionedInferenceBackendProtocol):
  def __init__(
    self,
    model_path: Path,
    inference_strategy: Literal["scores", "embeddings"],
    device_name: str,
    inference_library: LIBRARY_TYPES,
  ) -> None:
    in_idx = 0
    if inference_strategy == "scores":
      out_idx = 62
    elif inference_strategy == "embeddings":
      raise NotImplementedError(
        "Embeddings inference is not implemented for Geo TF models yet."
      )
    else:
      raise AssertionError()

    super().__init__(
      model_path,
      in_idx,
      out_idx,
      device_name,
      inference_library,
    )

  @classmethod
  def check_model_can_be_loaded(
    cls,
    model_path: Path,
    **kwargs: Any,
  ) -> int | None:
    n_outputs = check_tf_model_can_be_loaded(
      model_path=model_path, out_idx=62, **kwargs
    )
    return n_outputs


class PBGeoInferenceBackendV2_4(PBInferenceBackend2, VersionedInferenceBackendProtocol):
  def __init__(
    self,
    model_path: Path,
    inference_strategy: Literal["scores", "embeddings"],
    device_name: str,
  ) -> None:
    if inference_strategy == "scores":
      signature_name = "serving_default"
      prediction_key = "MNET_CLASS_ACTIVATION"
      input_key = "MNET_INPUT"
    elif inference_strategy == "embeddings":
      raise NotImplementedError(
        "Embeddings inference is not implemented for Geo PB models yet."
      )
    else:
      raise AssertionError()

    super().__init__(
      model_path,
      signature_name,
      prediction_key,
      input_key,
      device_name,
    )

  @classmethod
  def check_model_can_be_loaded(
    cls,
    model_path: Path,
    **kwargs: Any,
  ) -> int | None:
    n_outputs = check_pb_model_can_be_loaded(
      model_path,
      "serving_default",
      "MNET_CLASS_ACTIVATION",
    )
    return n_outputs


class GeoModelV2_4(GeoModelBase2):
  def __init__(
    self,
    model_path: Path,
    species_list: OrderedSet[str],
    use_custom_model: bool,
    backend_type: type[VersionedInferenceBackendProtocol],
    backend_custom_kwargs: dict[str, object] | None,
  ) -> None:
    super().__init__(model_path, species_list, use_custom_model)
    self._backend_type = backend_type
    self._backend_custom_kwargs = backend_custom_kwargs

  @classmethod
  def load(
    cls,
    model_path: Path,
    species_list: OrderedSet[str],
    backend_type: type[VersionedInferenceBackendProtocol],
    backend_custom_kwargs: dict[str, object] | None,
  ) -> GeoModelV2_4:
    result = GeoModelV2_4(
      model_path,
      species_list,
      use_custom_model=False,
      backend_type=backend_type,
      backend_custom_kwargs=backend_custom_kwargs,
    )
    return result

  @classmethod
  def load_custom(
    cls,
    model_path: Path,
    species_list: Path,
    backend_type: type[VersionedInferenceBackendProtocol],
    backend_custom_kwargs: dict[str, object] | None,
    check_validity: bool,
  ) -> GeoModelV2_4:
    assert model_path.exists()
    assert species_list.is_file()

    loaded_species_list: OrderedSet[str]
    try:
      loaded_species_list = get_species_from_file(species_list, encoding="utf8")
    except Exception as e:
      raise ValueError(
        f"Failed to read species list from '{species_list.absolute()}'. Ensure it is a valid text file."
      ) from e

    if check_validity:
      n_species_in_model = backend_type.check_model_can_be_loaded(
        model_path, **backend_custom_kwargs if backend_custom_kwargs is not None else {}
      )
      if n_species_in_model != len(loaded_species_list):
        raise ValueError(
          f"Model '{model_path.absolute()}' has {n_species_in_model} outputs, but species list '{species_list.absolute()}' has {len(loaded_species_list)} species!"
        )

    result = GeoModelV2_4(
      model_path,
      loaded_species_list,
      use_custom_model=True,
      backend_type=backend_type,
      backend_custom_kwargs=backend_custom_kwargs,
    )
    return result

  @classmethod
  @final
  def get_version(cls) -> GEO_MODEL_VERSIONS:
    return GEO_MODEL_VERSION_V2_4

  @classmethod
  @final
  def get_model_type(cls) -> MODEL_TYPES:
    return MODEL_TYPE_GEO

  def predict(
    self,
    latitude: float,
    longitude: float,
    /,
    *,
    week: int | None = None,
    min_confidence: float = 0.03,
    half_precision: bool = True,
    device: str = "CPU",
  ) -> PredictionResult:
    if not -90 <= latitude <= 90:
      raise ValueError(
        "Value for 'latitude' is invalid! It needs to be in interval [-90, 90]."
      )

    if not -180 <= longitude <= 180:
      raise ValueError(
        "Value for 'longitude' is invalid! It needs to be in interval [-180, 180]."
      )

    if not 0 <= min_confidence < 1.0:
      raise ValueError(
        "Value for 'min_confidence' is invalid! It needs to be in interval [0.0, 1.0)."
      )

    if week is not None and not (1 <= week <= 48):
      raise ValueError(
        "Value for 'week' is invalid! It needs to be either None or in interval [1, 48]."
      )

    if week is None:
      week = -1
    assert week is not None

    sample = np.expand_dims(np.array([latitude, longitude, week], dtype=np.float32), 0)

    prob_dtype: DTypeLike = np.float16 if half_precision else np.float32

    backend_loader = InferenceBackendLoader2(
      model_path=self.model_path,
      inference_strategy="scores",
      backend_type=self._backend_type,
      backend_custom_kwargs=self._backend_custom_kwargs,
    )

    backend = backend_loader.load_backend(device)
    res = backend.infer(sample)
    assert res.dtype == np.float32
    res = res.astype(prob_dtype, copy=False)

    res = np.squeeze(res, axis=0)

    species_ids = np.arange(
      len(self.species_list),
      dtype=uint_dtype_for(
        max(0, len(self.species_list) - 1),
      ),
    )

    invalid_mask = res < min_confidence
    prediction = PredictionResult(
      species_list=self.species_list,
      species_probs=res,
      species_ids=species_ids,
      species_masked=invalid_mask,
    )

    return prediction
