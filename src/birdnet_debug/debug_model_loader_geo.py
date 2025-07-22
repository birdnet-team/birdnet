import logging
import random
import sys
from multiprocessing import set_start_method
from pathlib import Path

import numpy as np

from birdnet.acoustic_models.inference.prediction_result import PredictionResult
from birdnet.logging_utils import get_package_logger
from birdnet.model_loader import (
  load,
  load_custom,
)
from birdnet_debug.hsn_downloader import get_hsn_file_paths
from birdnet_debug.pow_downloader import get_pow_file_paths

if __name__ == "__main__":
  # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # sämtliche TF-Logs

  import faulthandler
  import signal

  faulthandler.enable(file=sys.stderr, all_threads=True)
  faulthandler.register(signal.SIGUSR1)
  logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  root = get_package_logger()
  root.setLevel(logging.DEBUG)

  random.seed(0)
  np.random.seed(0)
  # tf.config.experimental.enable_op_determinism()
  import time

  set_start_method("fork", force=True)  # Linux, macOS

  start = time.perf_counter()
  backend = "tf"
  if backend == "tf":
    model = load("geo", "2.4", "tf", precision="fp32")
    # model = load_custom(
    #   "acoustic",
    #   "2.4",
    #   "tf",
    #   "/home/stefan/.local/share/birdnet/acoustic-models/v2.4/tf/model-fp32.tflite",
    #   "/home/stefan/.local/share/birdnet/acoustic-models/v2.4/tf/labels/en_us.txt",
    #   precision="fp32",
    #   check_validity=True,
    # )

    # model = load_custom(
    #   "/home/stefan/.local/share/birdnet/acoustic-models/v2.4/tf/model-fp32.tflite",
    #   "/home/stefan/.local/share/birdnet/acoustic-models/v2.4/tf/labels/en_us.txt",
    #   model_type="acoustic",
    #   version="2.4",
    #   backend="tf",
    #   precision="fp32",
    # )
    result = model.predict(
      20,
      50,
      week=1,
      min_confidence=0.03,
      half_precision=True,
      inference_library="litert",
    )
  elif backend == "pb":
    model = load("geo", "2.4", "pb")
    # model = load_custom(
    #   "acoustic",
    #   "2.4",
    #   "pb",
    #   "/home/stefan/.local/share/birdnet/acoustic-models/v2.4/pb/model/",
    #   "/home/stefan/.local/share/birdnet/acoustic-models/v2.4/pb/labels/en_us.txt",
    #   precision="fp32",
    #   check_validity=True,
    # )

    result = model.predict(
      20,
      50,
      week=1,
      min_confidence=0.03,
      half_precision=True,
      device="CPU",
    )
  end = time.perf_counter()
  print(f"Finished analysis in {end - start:.2f} seconds.")
  import tempfile

  output_file = Path(tempfile.gettempdir()) / "predictions.npz"
  now = time.perf_counter()
  result.save(output_file)
  print(f"Saved to {output_file} in {time.perf_counter() - now:.2f} seconds.")

  # if True:
  #   result_loaded = PredictionResult.load(output_file)
  #   df = result.to_dataframe()
  #   # for file in audio_paths:
  #   #   file_df = result.get_file_results(file).to_dataframe()
  #   #   file_df.to_csv(..)
  #   if len(df.index) > 0:
  #     print(f"Mean: {df['confidence'].mean()}, Shape: {df.shape}")
