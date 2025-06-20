import logging
import random
from enum import Enum
from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf

from birdnet_v2.acoustic_models.v2_4.tf import AcousticTFModelV2_4


class ModelType(str, Enum):
  # kanonische Namen
  ACOUSTIC = "acoustic"
  GEO = "geo"


# ModelTypeLit = Literal["acoustic", "geo"]


def load_old(spec: str, lang_id: str = "en_us"):
  if spec in ("acoustic/v2.4+tf@cpu", "acoustic", "acoustic/v2.4"):
    result = AcousticTFModelV2_4(lang_id)
  else:
    raise NotImplementedError(f"Model spec '{spec}' is not implemented.")
  return result


def load2(
  model_type: ModelType = ModelType.ACOUSTIC,
  version: Literal["2.4"] = "2.4",
  device: Literal["cpu", "gpu"] = "cpu",
  lang_id: str = "en_us",
):
  if model_type == "acoustic" and version == "2.4" and device == "cpu":
    result = AcousticTFModelV2_4(lang_id)
  else:
    raise NotImplementedError(f"Model spec is not implemented.")
  return result


def load(
  model_type: Literal["acoustic", "geo"] = "acoustic",
  version: Literal["2.4"] = "2.4",
  device: Literal["cpu", "gpu"] = "cpu",
  lang_id: str = "en_us",
):
  if model_type == "acoustic" and version == "2.4" and device == "cpu":
    result = AcousticTFModelV2_4(lang_id)
  else:
    raise NotImplementedError(f"Model spec is not implemented.")
  return result


def load_custom(
  model_path: Path,
  species_list: Path,
  model_type: Literal["acoustic"] = "acoustic",
  version: Literal["2.4"] = "2.4",
  device: Literal["cpu", "gpu"] = "cpu",
):
  pass


if __name__ == "__main__":
  # faulthandler.enable(file=sys.stderr, all_threads=True)
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )
  # alle dateien im ordner
  # audio_paths = [audio_paths[2]]
  # audio_paths = audio_paths[:3]

  folder = Path("test-dataset/test_dataset_5x2min")
  audio_paths = list(sorted(folder.glob("*.wav")))

  audio_paths = [
    Path("test-dataset/test_dataset_4x60min/0.wav"),
    Path("test-dataset/test_dataset_4x60min/1.wav"),
    Path("test-dataset/test_dataset_4x60min/2.wav"),
    Path("test-dataset/test_dataset_4x60min/3.wav"),
  ]
  audio_paths = [Path("test-dataset/test_dataset_1x10min/0.wav")]

  random.seed(0)
  np.random.seed(0)
  tf.random.set_seed(0)
  # tf.config.experimental.enable_op_determinism()
  import time

  # Example usage
  # model = load("acoustic")
  # model = load("acoustic/v2.4")
  # model = load("geo/v2.4+tf@cpu")
  model = load("acoustic", "2.4", "cpu", "en_us")

  # model = load_custom_model("acoustic/v2.4+pb@cpu", custom_species_list="..")
  audio_paths = [Path("test-dataset/test_dataset_1x60min/0.wav")]

  audio_paths = [
    Path("test-dataset/test_dataset_4x60min/0.wav"),
    Path("test-dataset/test_dataset_4x60min/1.wav"),
    Path("test-dataset/test_dataset_4x60min/2.wav"),
    Path("test-dataset/test_dataset_4x60min/3.wav"),
  ]
  # model.use_custom_model(model_path, custom_species_list="..")
  audio_paths = [Path("example/soundscape.wav")]

  start = time.perf_counter()
  result = model.analyze(
    audio_paths,
    n_jobs=12,
    batch_size=1,
    n_slots_factor=2,
    apply_sigmoid=False,
    top_k=2,
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
  df.to_csv("/tmp/predictions.csv", index=False)
  # for file in audio_paths:
  #   file_df = result.get_file_results(file).to_dataframe()
  #   file_df.to_csv(..)
  if len(df.index) > 0:
    print(f"Mean: {df['confidence'].mean()}, Shape: {df.shape}")
  print(f"Finished analysis in {end - start:.2f} seconds.")
  print("/tmp/predictions.csv written.")
