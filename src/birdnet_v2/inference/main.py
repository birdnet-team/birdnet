# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import time
from collections.abc import Generator
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import soundfile as sf  # pip install soundfile
from numpy.lib.stride_tricks import as_strided

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from tensorflow.lite.python import interpreter as tflite

from birdnet_v2.inference.consumer import Consumer
from birdnet_v2.inference.producer import (
  Producer,  # type: ignore
  load_audio_in_chunks_with_overlap,
  shm_ring,
)
from birdnet_v2.inference.species_tensor import SpeciesTensor
from birdnet_v2.inference.worker import ChildWorker, Worker

# ---------------------------------------------------------------------------
WIN_SEC = 3.0  # window length in seconds
HOP_SEC = 1.0  # hop size (sec)
SR = 32_000  # sample rate expected by the model
WIN_SAMPLES = int(WIN_SEC * SR)
EMPTY_ID = 0xFFFF

# (file_idx, first_win_idx, batch[B, N])
FrameBatch = Tuple[int, int, np.ndarray]

WIN_SEC = 3.0
HOP_SEC = 1.0


def analyze(
  files: List[Path],
  model_path: Path,
) -> SpeciesTensor:
  n_species = 6522
  n_jobs: int = 4
  batch_size = 50
  n_slots = n_jobs * 2
  prod_queue = mp.Queue()
  sem_free = mp.Semaphore(n_slots)
  sem_fill = mp.Semaphore()
  chunk_duration_s = 3
  overlap_duration_s = 0
  target_sample_rate = 48000
  default_confidence_threshold = 0.01
  custom_confidence_thresholds: Optional[dict[int, float]] = {5: -np.inf}
  top_k: int = 5

  use_bandpass: bool = False
  bandpass_fmin: Optional[float] = None
  bandpass_fmax: Optional[float] = None
  chunk_duration_samples = int(chunk_duration_s * target_sample_rate)
  whitelist: Optional[Sequence[int]] = None

  apply_sigmoid = False
  sigmoid_sensitivity = None

  result = SpeciesTensor(len(files), n_chunks=1148, top_k=top_k)

  logger = logging.getLogger(__name__)

  species_map = [f"sp_{i}" for i in range(n_species)]

  valid = np.zeros(len(species_map), bool)
  if whitelist:
    valid[list(whitelist)] = True
  else:
    valid[:] = True
  valid.setflags(write=False)

  thresholds = np.full(n_species, default_confidence_threshold, np.float32)
  if custom_confidence_thresholds:
    for sp_id, threshold in custom_confidence_thresholds.items():
      if 0 <= sp_id < len(thresholds):
        thresholds[sp_id] = threshold

  with (
    shm_ring(
      "bnet_ring_file_indices",
      n_slots * batch_size * np.dtype(np.uint32).itemsize,
    ),
    shm_ring(
      "bnet_ring_chunk_indices", n_slots * batch_size * np.dtype(np.uint32).itemsize
    ),
    shm_ring(
      "bnet_ring_audio_samples",
      n_slots * batch_size * chunk_duration_samples * np.dtype(np.float32).itemsize,
    ),
  ):
    print("Shared memory initialized.")

    prod = mp.Process(
      target=Producer(
        files,
        batch_size,
        n_slots,
        n_jobs,
        prod_queue,
        sem_free,
        sem_fill,
        chunk_duration_s,
        overlap_duration_s,
        target_sample_rate,
        bandpass_fmax=bandpass_fmax,
        bandpass_fmin=bandpass_fmin,
        use_bandpass=use_bandpass,
        fmax=15000,
        fmin=0,
      ),
      daemon=True,
    )
    prod.start()

    worker_queue = mp.Queue()
    workers = [
      mp.Process(
        target=ChildWorker(
          model_path=model_path,
          thresh=thresholds,
          top_k=top_k,
          valid=valid,
          batch_size=batch_size,
          n_slots=n_slots,
          chunk_duration_samples=chunk_duration_samples,
          job_q=prod_queue,
          out_q=worker_queue,
          sem_fill=sem_fill,
          sem_free=sem_free,
          apply_sigmoid=apply_sigmoid,
          sigmoid_sensitivity=sigmoid_sensitivity,
        ),
        daemon=True,
      )
      for _ in range(n_jobs)
    ]

    for w in workers:
      w.start()

    consumer = Consumer(n_jobs, worker_queue, result)
    consumer()

    prod.join()
    logger.info("Producer finished.")

    for w in workers:
      w.join()
      logger.info(f"Worker {w.pid} finished.")
    logger.info("All workers finished.")

  return result


def test():
  audio_path, duration = Path("test-dataset/test_dataset_1x1440min/0.wav"), 1440
  audio_path, duration = Path("test-dataset/test_dataset_1x60min/0.wav"), 60
  audio_path, duration = Path("example/soundscape.wav"), 2

  n_files = 1
  paths = [audio_path] * n_files
  model_path = Path(
    "/home/stefan/.local/share/birdnet/models/v2.4/TFLite/audio-model.tflite"
  )
  model_path = Path(
    "src/birdnet_legacy/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
  )

  from time import perf_counter

  start = perf_counter()
  print(f"Started analysis... {time.strftime('%H:%M:%S')}")
  result = analyze(
    paths,
    model_path,
  )
  end = perf_counter()
  print(f"Finished analysis in {end - start:.2f} seconds.")
  print(f"Finished analysis in {(end - start) / n_files:.2f} seconds per file.")
  print(f"Finished analysis in {(end - start) / n_files / duration * 60:.2f} s/h.")
  print(f"Finished analysis in {(end - start) / n_files / duration * 1000:.2f} ms/min.")
  # print(result._species_probs.tolist())


if __name__ == "__main__":
  import faulthandler
  import sys

  # faulthandler.enable(file=sys.stderr, all_threads=True)
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )
  test()
