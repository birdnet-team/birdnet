import logging
import random
from pathlib import Path
from typing import Literal

import numpy as np

from birdnet_v2.acoustic_models.v2_4.tf import AcousticTFModelV2_4


def load(spec: str = "acoustic", lang_id: str = "en_us"):
  if spec in ("acoustic/v2.4+tf@cpu", "acoustic", "acoustic/v2.4"):
    result = AcousticTFModelV2_4(lang_id)
  else:
    raise NotImplementedError(f"Model spec '{spec}' is not implemented.")
  return result


import tensorflow as tf

if __name__ == "__main__":
  # faulthandler.enable(file=sys.stderr, all_threads=True)
  logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )
  # Example usage
  model = load("acoustic/v2.4+tf@cpu")

  # alle dateien im ordner
  folder = Path("test-dataset/test_dataset_5x2min")
  audio_paths = list(sorted(folder.glob("*.wav")))
  # audio_paths = [audio_paths[2]]
  #audio_paths = audio_paths[:3]

  audio_paths = [Path("test-dataset/test_dataset_1x60min/0.wav")]
  audio_paths = [Path("example/soundscape.wav")]
  audio_paths = [
    Path("test-dataset/test_dataset_4x60min/0.wav"),
    #Path("test-dataset/test_dataset_4x60min/1.wav"),
    # Path("test-dataset/test_dataset_4x60min/2.wav"),
    # Path("test-dataset/test_dataset_4x60min/3.wav"),
  ]

  tf.random.set_seed(0)
  random.seed(0)
  np.random.seed(0)
  tf.config.experimental.enable_op_determinism()

  result = model.analyze(
    audio_paths,
    n_jobs=1,
    batch_size=4,
    n_slots_factor=4,
    apply_sigmoid=False,
    top_k=1,
    overlap_duration_s=0,
    sigmoid_sensitivity=1,
    default_confidence_threshold=-np.inf,
    # custom_species_list={
    #   "Junco hyemalis_Dark-eyed Junco",
    #   "Haemorhous mexicanus_House Finch",
    # },
  )
  result.to_csv("/tmp/predictions.csv", index=False)
  print(result)
  print("/tmp/predictions.csv written.")
