import platform
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from birdnet.acoustic_models.inference.scores.prediction_result import PredictionResult
from birdnet.acoustic_models.v2_4.model import AcousticModelV2_4
from birdnet.backends import litert_installed
from birdnet.helper import get_hash
from birdnet.local_data import get_package_version
from birdnet.model_loader import load
from birdnet_tests.helper import tensorflow_gpu_available
from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT

JOINT_REPORT = Path("benchmarks/consistency/report.csv")
COMPARE_THRESHOLDS = [0, 0.001, 0.01, 0.1, 0.2, 0.3]
TEST_FILE = TEST_FILE_SHORT
TEST_FILE = TEST_FILE_LONG
assert TEST_FILE.is_file()


def predict(
  model: AcousticModelV2_4,
  device: str,
  n_workers: int | None,
  batch_size: int,
) -> dict:
  start = perf_counter()
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
    init_time = perf_counter() - start
    result = session.run(TEST_FILE)
  total_time = perf_counter() - start
  prediction_result = {
    "backend": model._backend_type.name(),
    "precision": model._backend_type.precision(),
    "init_time_s": init_time,
    "prediction_time_s": total_time - init_time,
    "total_time_s": total_time,
    "device": device,
    "n_workers": n_workers,
    "batch_size": batch_size,
    "result": result,
  }
  return prediction_result


def run_reference_model():
  pb_model = load("acoustic", "2.4", "pb", precision="fp32")

  pb_cpu = predict(pb_model, "CPU", n_workers=None, batch_size=1)
  print("PB CPU done.")
  return pb_cpu


def run_gpu_model() -> None:
  if tensorflow_gpu_available():
    reference = run_reference_model()
    pb_model = load("acoustic", "2.4", "pb", precision="fp32")
    pb_gpu = predict(pb_model, "GPU", n_workers=1, batch_size=1025)
    print("PB GPU done.")
    create_report(reference, [pb_gpu], COMPARE_THRESHOLDS)


def get_sorted_probs(prediction_result: PredictionResult) -> np.ndarray:
  sort_idx = np.argsort(prediction_result.species_ids, axis=-1)
  sorted_probs = np.take_along_axis(prediction_result.species_probs, sort_idx, axis=-1)
  return sorted_probs


def create_report(reference, results, thresholds: list[float]):
  report = []

  now = datetime.now()
  now_time = now.strftime("%Y/%m/%d %I:%M:%S %p")
  now_fname = now.strftime("%Y-%m-%d_%H%M%S")

  meta = {
    "setup_hash": "",
    "version": get_package_version(),
    "python": f"{platform.python_version()} {platform.python_implementation()}",
    "hw_host": platform.platform(),
    "hw_cpu": platform.processor(),
  }
  platform_hash = f"{meta['python']}-{meta['hw_host']}-{meta['hw_cpu']}"
  hash_digest = get_hash(platform_hash)[:6]
  meta["setup_hash"] = hash_digest

  fname = "{}"

  ref = get_sorted_probs(reference["result"])
  for res in results:
    for threshold in thresholds:
      mask = ref >= threshold
      scores = get_sorted_probs(res["result"])
      prob_diff = np.abs(ref - scores)
      masked_diff = prob_diff[mask]
      max_diff = np.max(masked_diff)
      mean_diff = np.mean(masked_diff)
      min_diff = np.min(masked_diff)
      std_diff = np.std(masked_diff)
      q1 = np.percentile(masked_diff, 25)
      median_diff = np.median(masked_diff)
      q3 = np.percentile(masked_diff, 75)
      n_segments = ref.shape[1]
      n_species = ref.shape[2]
      n_values = masked_diff.size
      total_values = prob_diff.size

      report_entry = meta | {
        "date": now_time,
        "backend": res["backend"],
        "precision": res["precision"],
        "n_species": n_species,
        "device": res["device"],
        # "init_time_s": res["init_time_s"],
        # "prediction_time_s": res["prediction_time_s"],
        # "total_time_s": res["total_time_s"],
        "compare_threshold": threshold,
        "n_segments": n_segments,
        "total_values": total_values,
        "n_values": n_values,
        "n_values_percent": round(n_values / total_values * 100, 2),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "min_diff": min_diff,
        "max_diff": max_diff,
        "q1_diff": q1,
        "q2_diff": median_diff,
        "q3_diff": q3,
      }
      report.append(report_entry)
  df_report = pd.DataFrame(report)
  return df_report


def run_tflite_models():
  reference = run_reference_model()

  model_tf32 = load("acoustic", "2.4", "tf", precision="fp32", library="tf")
  tf32 = predict(model_tf32, "CPU", n_workers=None, batch_size=1)
  print("TF FP32 done.")

  model_tf16 = load("acoustic", "2.4", "tf", precision="fp16", library="tf")
  tf16 = predict(model_tf16, "CPU", n_workers=None, batch_size=1)
  print("TF FP16 done.")

  model_int8 = load("acoustic", "2.4", "tf", precision="int8", library="tf")
  int8 = predict(model_int8, "CPU", n_workers=None, batch_size=1)
  print("TF INT8 done.")
  create_report(reference, [tf32, tf16, int8], COMPARE_THRESHOLDS)


def run_litert_tests():
  if litert_installed():
    reference = run_reference_model()

    model_tf32 = load("acoustic", "2.4", "tf", precision="fp32", library="litert")
    tf32 = predict(model_tf32, "CPU", n_workers=None, batch_size=1)
    print("LiteRT FP32 done.")

    model_tf16 = load("acoustic", "2.4", "tf", precision="fp16", library="litert")
    tf16 = predict(model_tf16, "CPU", n_workers=None, batch_size=1)
    print("LiteRT FP16 done.")

    model_int8 = load("acoustic", "2.4", "tf", precision="int8", library="litert")
    int8 = predict(model_int8, "CPU", n_workers=None, batch_size=1)
    print("LiteRT INT8 done.")
    create_report(reference, [tf32, tf16, int8], COMPARE_THRESHOLDS)


def merge_results():
  df_report = pd.DataFrame()
  for file in JOINT_REPORT.parent.glob("*.csv"):
    df_part = pd.read_csv(file)
    df_report = pd.concat([df_report, df_part], ignore_index=True)
  return df_report


def main():
  run_gpu_model()
  run_litert_tests()
  run_tflite_models()


if __name__ == "__main__":
  df_report.to_csv("consistency_report.csv", index=False)
