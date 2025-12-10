import importlib.metadata
import multiprocessing as mp
import pickle
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.backends import litert_installed
from birdnet.local_data import get_package_version
from birdnet.model_loader import load


def _check_tf_gpu() -> bool:
  try:
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    return len(devices) > 0
  except Exception:
    return False


def tensorflow_gpu_available() -> bool:
  ctx = mp.get_context()
  with ctx.Pool(1) as pool:
    result = pool.apply(_check_tf_gpu)
  return result


def predict(
  run_name: str,
  model: AcousticModelV2_4,
  device: str,
  n_workers: int | None,
  batch_size: int,
) -> dict:
  path = Path("consistency_test.wav")
  if not path.is_file():
    raise ValueError(
      f"Test file not found. Ensure the test file is available: {path.absolute()}"
    )

  with model.predict_session(
    top_k=None,
    default_confidence_threshold=-np.inf,
    n_workers=n_workers,
    n_feeders=1,
    prefetch_ratio=1,
    half_precision=False,
    show_stats=None,
    max_audio_duration_min=None,
    device=device,
    max_n_files=1,
    batch_size=batch_size,
    overlap_duration_s=2.0,
    bandpass_fmin=0,
    bandpass_fmax=15_000,
    apply_sigmoid=True,
    sigmoid_sensitivity=1.0,
    custom_species_list=None,
  ) as session:
    result = session.run(path)
  prediction_result = {
    "run_name": run_name,
    "backend": model._backend_type.name(),
    "precision": model._backend_type.precision(),
    "device": device,
    "result": result,
    "version": get_package_version(),
    "python": f"{platform.python_version()} {platform.python_implementation()}",
    "hw_host": platform.platform(),
    "hw_cpu": platform.processor(),
    "tensorflow_version": importlib.metadata.version("tensorflow"),
  }

  return prediction_result


def run_reference_model():
  pb_model = load("acoustic", "2.4", "pb", precision="fp32")

  pb_cpu = predict("pb-cpu", pb_model, "CPU", n_workers=None, batch_size=1)
  print("PB CPU done.")
  return pb_cpu


def run_gpu_model() -> list[dict]:
  assert tensorflow_gpu_available()
  pb_model = load("acoustic", "2.4", "pb", precision="fp32")
  pb_gpu = predict("pb-gpu", pb_model, "GPU", n_workers=1, batch_size=1025)
  print("PB GPU done.")
  return [pb_gpu]


def run_tflite_models() -> list[dict]:
  run_name = "tflite"

  model_tf32 = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  tf32 = predict(run_name, model_tf32, "CPU", n_workers=None, batch_size=1)
  print("TF FP32 done.")

  model_tf16 = load("acoustic", "2.4", "tf", precision="fp16", library="tf")
  tf16 = predict(run_name, model_tf16, "CPU", n_workers=None, batch_size=1)
  print("TF FP16 done.")

  model_int8 = load("acoustic", "2.4", "tf", precision="int8", library="tf")
  int8 = predict(run_name, model_int8, "CPU", n_workers=None, batch_size=1)
  print("TF INT8 done.")
  return [tf32, tf16, int8]


def run_litert_tests() -> list[dict]:
  assert litert_installed()

  run_name = "litert"

  model_tf32 = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
  tf32 = predict(run_name, model_tf32, "CPU", n_workers=None, batch_size=1)
  print("LiteRT FP32 done.")

  model_tf16 = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
  tf16 = predict(run_name, model_tf16, "CPU", n_workers=None, batch_size=1)
  print("LiteRT FP16 done.")

  model_int8 = load("acoustic", "2.4", "tf", precision="int8", library="litert")
  int8 = predict(run_name, model_int8, "CPU", n_workers=None, batch_size=1)
  print("LiteRT INT8 done.")
  return [tf32, tf16, int8]


def save_results(results: list[dict]):
  now = datetime.now()
  now_fname = now.strftime("%Y-%m-%d_%H-%M-%S")

  fname = f"{now_fname}_report.pkl"
  out_path = Path(__file__).parent
  report_path = out_path / fname
  report_path.parent.mkdir(parents=True, exist_ok=True)

  with open(report_path, "wb") as f:
    pickle.dump(results, f)
  print("Saved results to:", report_path.absolute())
  print(
    "Please include a name (e.g., '..._report_stefan_laptop') and update the results to: https://mytuc.org/sknk"
  )


def main() -> None:
  reports = []
  reference = run_reference_model()
  reports.append(reference)
  if tensorflow_gpu_available():
    reports.extend(run_gpu_model())
  # run is possible only in this order
  if litert_installed():
    reports.extend(run_litert_tests())
  reports.extend(run_tflite_models())

  save_results(reports)


if __name__ == "__main__":
  main()
