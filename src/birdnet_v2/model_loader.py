import logging
import random
from multiprocessing import set_start_method
from pathlib import Path
from typing import Literal, overload

import numpy as np
import tensorflow as tf

from birdnet_v2.acoustic_models.v2_4.pb import AcousticPBModelV2_4
from birdnet_v2.acoustic_models.v2_4.tf import AcousticTFModelV2_4
from birdnet_v2.base import (
  MODEL_BACKEND_PB,
  MODEL_BACKEND_TF,
  MODEL_BACKENDS,
  MODEL_TYPE_ACOUSTIC,
  MODEL_TYPE_GEO,
  MODEL_TYPES,
  MODEL_VERSION_V2_4,
  MODEL_VERSIONS,
)
from birdnet_v2.logging_utils import get_package_logger
from birdnet_v2_tests.hsn_downloader import get_hsn_file_paths
from birdnet_v2_tests.pow_downloader import get_pow_file_paths

# models: list[ModelBase] = [
#   AcousticTFModelV2_4,
# ]


@overload
def load(  # type: ignore
  *,
  model_type: Literal["acoustic"] = ...,
  version: Literal["2.4"] = ...,
  backend: Literal["tf"] = ...,
  device: Literal["cpu", "gpu"] = ...,
  lang_id: str = ...,
) -> AcousticTFModelV2_4: ...


@overload
def load(
  *,
  model_type: Literal["acoustic"] = ...,
  version: Literal["2.4"] = ...,
  backend: Literal["pb"] = ...,
  device: Literal["cpu", "gpu"] = ...,
  lang_id: str = ...,
) -> AcousticPBModelV2_4: ...


# @overload
# def load(
#   *,
#   model_type: Literal["acoustic"] = MODEL_TYPE_ACOUSTIC,
#   version: Literal["2.4"] = MODEL_VERSION_V2_4,
#   backend: Literal["tf"] = MODEL_BACKEND_TF,
#   device: Literal["cpu", "gpu"] = "cpu",
#   lang_id: str = "en_us",
# ) -> AcousticTFModelV2_4: ...


# @overload
# def load(
#   *,
#   model_type: Literal["acoustic"] = MODEL_TYPE_ACOUSTIC,
#   version: Literal["2.4"] = MODEL_VERSION_V2_4,
#   backend: Literal["pb"] = MODEL_BACKEND_PB,
#   device: Literal["cpu", "gpu"] = "cpu",
#   lang_id: str = "en_us",
# ) -> AcousticPBModelV2_4: ...


def load(
  *,
  model_type: MODEL_TYPES = MODEL_TYPE_ACOUSTIC,
  version: MODEL_VERSIONS = MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  device: str = "CPU",
  lang_id: str = "en_us",
):
  all_devices_with_name = [
    log_dev
    for log_dev in tf.config.list_logical_devices()
    if device.lower() in log_dev.name.lower()
  ]
  if len(all_devices_with_name) == 0:
    raise Exception("No CPU found!")
  logical_device = all_devices_with_name[0]
  if len(all_devices_with_name) > 1:
    logging.warning(
      f"Multiple devices found: {all_devices_with_name}. "
      f"Using the first one ('{logical_device.name}')."
    )

  if model_type == MODEL_TYPE_ACOUSTIC:
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        if logical_device.device_type != "CPU":
          raise ValueError("TF models can only be loaded on CPU!")
        return AcousticTFModelV2_4.load_official(lang_id)
      else:
        assert backend == MODEL_BACKEND_PB
        return AcousticPBModelV2_4.load_official(lang_id, logical_device.name)
    raise AssertionError()
  else:
    assert model_type == MODEL_TYPE_GEO
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        pass
      else:
        assert backend == MODEL_BACKEND_PB
    raise AssertionError()


def load_custom(
  model_path: Path,
  species_list: Path,
  model_type: MODEL_TYPES = MODEL_TYPE_ACOUSTIC,
  version: MODEL_VERSIONS = MODEL_VERSION_V2_4,
  backend: MODEL_BACKENDS = MODEL_BACKEND_TF,
  device: Literal["cpu", "gpu"] = "cpu",
):
  if model_type == MODEL_TYPE_ACOUSTIC:
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        if device != "cpu":
          raise ValueError("TF models can only be loaded on CPU!")
        return AcousticTFModelV2_4.load_custom(model_path, species_list)
      else:
        assert backend == MODEL_BACKEND_PB
    raise AssertionError()
  else:
    assert model_type == MODEL_TYPE_GEO
    if version == MODEL_VERSION_V2_4:
      if backend == MODEL_BACKEND_TF:
        pass
      else:
        assert backend == MODEL_BACKEND_PB
    raise AssertionError()


if __name__ == "__main__":
  # set_start_method("forkserver", force=True) # Linux, macOS
  # set_start_method("spawn", force=True)  # Windows
  set_start_method("fork", force=True)  # Linux, macOS

  # faulthandler.enable(file=sys.stderr, all_threads=True)
  logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  root = get_package_logger()
  root.setLevel(logging.DEBUG)

  folder = Path("test-dataset/test_dataset_5x2min")
  audio_paths = list(sorted(folder.glob("*.wav")))

  audio_paths = [
    Path("test-dataset/test_dataset_4x60min/0.wav"),
    Path("test-dataset/test_dataset_4x60min/1.wav"),
    Path("test-dataset/test_dataset_4x60min/2.wav"),
    Path("test-dataset/test_dataset_4x60min/3.wav"),
  ]

  random.seed(0)
  np.random.seed(0)
  tf.random.set_seed(0)
  # tf.config.experimental.enable_op_determinism()
  import time

  # Example usage
  # model = load("acoustic")
  # model = load("acoustic/v2.4")
  # model = load("geo/v2.4+tf@cpu")
  model = load()
  model = load(backend="pb", device="cpu", lang_id="de")
  model = load(device="cpu", lang_id="de")
  model = load(backend="pb")
  # model = load_custom_model("acoustic/v2.4+pb@cpu", custom_species_list="..")

  # model.use_custom_model(model_path, custom_species_list="..")
  audio_paths = [Path("src/birdnet_v2_debug/10min.wav")]
  audio_paths = [Path("src/birdnet_v2_debug/60min.wav")]

  audio_paths = [Path("test-dataset/test_dataset_1x10min/0.wav")]

  audio_paths = [
    Path("test-dataset/test_dataset_4x60min/0.wav"),
    Path("test-dataset/test_dataset_4x60min/1.wav"),
    Path("test-dataset/test_dataset_4x60min/2.wav"),
    Path("test-dataset/test_dataset_4x60min/3.wav"),
  ]
  audio_paths = [
    Path("src/birdnet_tests/test_files/soundscape.wav"),
    # Path("src/birdnet_tests/test_files/soundscape.flac"),
  ]
  audio_paths = [Path("example/soundscape.wav")]
  audio_paths = get_pow_file_paths()
  audio_paths = [Path("test-dataset/test_dataset_1x60min/0.wav")]
  audio_paths = get_hsn_file_paths()
  start = time.perf_counter()
  result = model.analyze(
    audio_paths,
    n_jobs=1,
    n_prods=2,
    batch_size=1,
    n_slots_factor=2,
    apply_sigmoid=False,
    top_k=5,
    overlap_duration_s=0,
    sigmoid_sensitivity=1,
    default_confidence_threshold=-np.inf,
    track_performance=True,
    half_precision=True,
    custom_confidence_thresholds={
      "Junco hyemalis_Dark-eyed Junco": -np.inf,
      "Haemorhous mexicanus_House Finch": 0.1,
    },
    # custom_confidence_thresholds={
    #   model.species_list.by_scientific_name("Junco hyemalis"): -np.inf,
    #   model.species_list.by_common_name("House Finch"): -np.inf,
    # },
    # max_audio_duration_min=60,
    # custom_species_list={
    #   "Junco hyemalis_Dark-eyed Junco",
    #   "Haemorhous mexicanus_House Finch",
    # },
  )
  end = time.perf_counter()
  df = result.to_dataframe()
  import tempfile

  output_file = Path(tempfile.gettempdir()) / "predictions.csv"
  df.to_csv(output_file, index=False)
  # for file in audio_paths:
  #   file_df = result.get_file_results(file).to_dataframe()
  #   file_df.to_csv(..)
  if len(df.index) > 0:
    print(f"Mean: {df['confidence'].mean()}, Shape: {df.shape}")
  print(f"Finished analysis in {end - start:.2f} seconds.")
  print(f"{output_file.absolute()} written.")
