# birdnet_batch_inference.py – raw‑audio version
"""Ultra‑light multiprocessing pipeline for BirdNET TFLite models where the
**direct model input is PCM audio windows**, *not* pre‑computed mels.

* Windows: 3‑s mono @ 32 kHz  ⇒  96 000 samples each (by default).
* Hop:     1 s (overlap 66 %).
* Producer reads \`chunk_sec\` audio from disk, slices it into windows and
  queues individual windows.
* Worker collects \`batch_frames\` windows, runs **one** `tflite.invoke`, does
  Top‑k + threshold + whitelist masking and returns the block.
* Writer owns the contiguous Top‑k tensor and resizes it *linearly* (exact new
  length) when necessary.

This version omits all mel‑spectrogram code and relies on **soundfile** (libsndfile)
for PCM decoding. Replace the read helper if you favour ffmpeg or another
backend.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import sys
import time
from collections.abc import Generator
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

from birdnet_v2.consumer import Consumer, SpeciesTensor
from birdnet_v2.producer import (
  Producer,  # type: ignore
  load_audio_in_chunks_with_overlap,
  shm_ring,
)
from birdnet_v2.worker import Worker

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
  n_jobs: int = 24
  batch_size = 50
  n_slots = n_jobs * 2
  prod_queue = mp.Queue()
  sem_free = mp.Semaphore(n_slots)
  sem_fill = mp.Semaphore()
  chunk_duration_s = 3
  overlap_duration_s = 0
  target_sample_rate = 48000
  chunk_duration_samples = int(chunk_duration_s * target_sample_rate)

  with (
    shm_ring(
      "bnet_ring_file_indices",
      n_slots * batch_size * np.dtype(np.uint32).itemsize,
    ) as shm_file_indices,
    shm_ring(
      "bnet_ring_chunk_indices", n_slots * batch_size * np.dtype(np.uint32).itemsize
    ) as shm_chunk_indices,
    shm_ring(
      "bnet_ring_audio_samples",
      n_slots * batch_size * chunk_duration_samples * np.dtype(np.float32).itemsize,
    ) as shm_audio_samples,
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
      ),
      daemon=True,
    )
    prod.start()

    worker = Worker(
      model_path=model_path,
      prod_queue=prod_queue,
      batch_size=batch_size,
      n_slots=n_slots,
      chunk_duration_samples=chunk_duration_samples,
      sem_free=sem_free,
      sem_fill=sem_fill,
      n_jobs=n_jobs,
      threshold=0.01,
      top_k=5,
      whitelist=None,
    )

    worker.start()

    consumer = Consumer(n_files=len(files), worker=worker, init_w=1148)
    tensor = consumer.consume()

    prod.join()
    print("Producer finished.")
    worker.join()
    print("Workers finished.")

  print("Producer finished processing.")
  return tensor


def test():
  audio_path, duration = Path("test-dataset/test_dataset_1x1440min/0.wav"), 1440
  audio_path, duration = Path("example/soundscape.wav"), 2
  audio_path, duration = Path("test-dataset/test_dataset_1x60min/0.wav"), 60
  
  n_files = 100
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
  print(result._species_probs.tolist())


if __name__ == "__main__":
  import faulthandler
  import sys

  # faulthandler.enable(file=sys.stderr, all_threads=True)

  test()
