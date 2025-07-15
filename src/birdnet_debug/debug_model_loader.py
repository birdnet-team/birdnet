import logging
import random
import sys
from multiprocessing import set_start_method
from pathlib import Path

import numpy as np

from birdnet.acoustic_models.inference.prediction_result import PredictionResult
from birdnet.acoustic_models.v2_4.base import AcousticModelBaseV2_4
from birdnet.logging_utils import get_package_logger
from birdnet.model_loader import load
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
  # tf.config.experimental.enable_op_determinism()
  import time

  # import absl.logging as absl_logging

  # absl_logging.set_verbosity(absl_logging.ERROR)  # absl-Backend
  # absl_logging.set_stderrthreshold("error")

  # Example usage
  # model = load("acoustic")
  # model = load("acoustic/v2.4")
  # model = load("geo/v2.4+tf@cpu")
  # os.environ["CUDA_VISIBLE_DEVICES"] = ""
  # 1) Device-Logs deaktivieren
  # tf.debugging.set_log_device_placement(False)
  # 2) C++-Logger auf WARN oder ERROR stellen
  # 0=alle, 1=INFO, 2=WARNING, 3=ERROR
  # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

  # logging.getLogger("tensorflow").setLevel(logging.ERROR)
  # logging.getLogger("tensorflow").propagate = False
  # warnings.filterwarnings(
  #   "ignore",
  #   message=r".*Importing a function .*unsaved custom gradients.*",
  #   category=UserWarning,
  # )

  # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  # tf.get_logger().setLevel("WARNING")        # Python-Logger ebenfalls drosseln

  # model = load_custom_model("acoustic/v2.4+pb@cpu", custom_species_list="..")

  # model.use_custom_model(model_path, custom_species_list="..")
  audio_paths = [Path("src/birdnet_v2_debug/60min.wav")]

  audio_paths = [
    Path("src/birdnet_tests/test_files/soundscape.wav"),
    # Path("src/birdnet_tests/test_files/soundscape.flac"),
  ]

  """Gibt eine Liste der Pfade zu den Dateien im Zielverzeichnis zurück."""
  audio_paths = [Path("example/soundscape.wav")]
  audio_paths = get_pow_file_paths()
  audio_paths.extend(get_hsn_file_paths())
  audio_paths.extend(list(Path("test-dataset/HSN copy").glob("**/*.flac")))
  audio_paths.extend(list(Path("test-dataset/LARGE").glob("**/*.wav")))
  audio_paths.extend(list(Path("test-dataset/LARGE").glob("**/*.WAV")))

  audio_paths = list(Path("test-dataset/test_dataset_100x60min").glob("**/*.wav"))
  audio_paths = list(Path("test-dataset/test_dataset_200x60min").glob("**/*.wav"))
  audio_paths = list(
    Path("/home/mi/sttau/test-datasets/test_dataset_1000x60min").glob("**/*.wav")
  )
  params_1000h_3gpu = {
    "n_workers": 3,
    "n_producers": 10,
    "batch_size": 1000,
    "prefetch_ratio": 4,
  }

  audio_paths = list(
    Path("/home/mi/sttau/test-datasets/test_dataset_100x60min").glob("**/*.wav")
  )
  params_100h_48cpu = {
    "n_workers": 45,
    "n_producers": 3,
    "batch_size": 1,
    "prefetch_ratio": 2,
  }
  res_100h_4cpu = "inference speed: 69 ms/chunk; 636 chunks/s; 31.81 min/s; memory usage: 27010.21 MiB; CPU usage: 50.7%; prel: 45; free: 0; busy: 45; fill: 90; progress: 99.53%; remaining: 0:00:01"
  params = params_100h_48cpu
  audio_paths = list(
    Path("/home/mi/sttau/test-datasets/test_dataset_1000x60min").glob("**/*.wav")
  )
  params_1000h_4cpu = {
    "n_workers": 3,
    "n_producers": 1,
    "batch_size": 1,
    "prefetch_ratio": 4,
  }

  params_1000h_48cpu = {
    "n_workers": 45,
    "n_producers": 3,
    "batch_size": 1,
    "prefetch_ratio": 4,
  }
  params = params_1000h_48cpu

  params = {
    "n_workers": 1,
    "n_producers": 3,
    "batch_size": 1,
    "prefetch_ratio": 2,
    "backend": "tf",
    "device": "CPU",
  }
  # model = load(device="CPU", lang_id="de")
  # model = load(backend="pb", device="gpu:0")
  # model = load()

  audio_paths = Path("src/birdnet_tests/test_files/soundscape.wav")

  audio_paths = [
    Path("src/birdnet_tests/test_files/soundscape.wav"),
    Path("test-dataset/test_dataset_1x60min/0.wav"),
  ]
  audio_paths = [Path("src\\birdnet_v2_debug\\10min.wav")]
  audio_paths = [Path("src\\birdnet_v2_debug\\60min.wav")]

  audio_paths = "test-dataset/test_dataset_10000x0.2s_flac"
  audio_paths = "test-dataset/test_dataset_1000x0.2s_flac"
  audio_paths = "test-dataset/test_dataset_100x1.3s_flac"
  audio_paths = "test-dataset/test_dataset_100x1.3s_flac/000.flac"
  audio_paths = "test-dataset/test_dataset_1x7.3s_flac/0.flac"

  audio_paths = "test-dataset/test_dataset_1x10min/0.wav"
  audio_paths = "test-dataset/test_dataset_100000x4s_flac"

  audio_paths = "test-dataset/test_dataset_4x60min/0.wav"

  audio_paths = "example/soundscape.wav"

  audio_paths = [
    Path("test-dataset/test_dataset_4x60min/0.wav"),
    Path("test-dataset/test_dataset_4x60min/1.wav"),
    Path("test-dataset/test_dataset_4x60min/2.wav"),
    Path("test-dataset/test_dataset_4x60min/3.wav"),
  ]
  audio_paths = get_pow_file_paths()
  params = {
    "n_workers": 12,
    "n_producers": 1,
    "batch_size": 1,
    "prefetch_ratio": 2,
    "backend": "tf",
    "device": "CPU",
  }

  model: AcousticModelBaseV2_4 = load(backend=params["backend"])

  # set_start_method("forkserver", force=True) # Linux, macOS
  # set_start_method("spawn", force=True)  # Linux, macOS
  set_start_method("fork", force=True)  # Linux, macOS
  start = time.perf_counter()
  assert isinstance(model, AcousticModelBaseV2_4)
  result = model.analyze(
    audio_paths,
    workers=params["n_workers"],
    feeders=params["n_producers"],
    batch_size=params["batch_size"],
    prefetch_ratio=params["prefetch_ratio"],
    apply_sigmoid=True,
    top_k=1,
    overlap_duration_s=0,
    sigmoid_sensitivity=1,
    default_confidence_threshold=-np.inf,
    half_precision=False,
    device=params["device"],
    show_stats="benchmark",
    serial_io=False,
    # custom_confidence_thresholds={
    #   "Junco hyemalis_Dark-eyed Junco": -np.inf,
    #   "Haemorhous mexicanus_House Finch": 0.1,
    # },
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
  print(f"Finished analysis in {end - start:.2f} seconds.")
  import tempfile

  output_file = Path(tempfile.gettempdir()) / "predictions.npz"
  now = time.perf_counter()
  result.dump(output_file)
  print(f"Saved to {output_file} in {time.perf_counter() - now:.2f} seconds.")
  if True:
    result_loaded = PredictionResult.load(output_file)
    df = result.to_dataframe()
    # for file in audio_paths:
    #   file_df = result.get_file_results(file).to_dataframe()
    #   file_df.to_csv(..)
    if len(df.index) > 0:
      print(f"Mean: {df['confidence'].mean()}, Shape: {df.shape}")
