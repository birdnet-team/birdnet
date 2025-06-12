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

from birdnet_v3.consumer import Consumer, SpeciesTensor
from birdnet_v3.producer import (
  Producer,  # type: ignore
  load_audio_in_chunks_with_overlap,
)
from birdnet_v3.worker import Worker

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
  *,
  model_path: Path,
) -> SpeciesTensor:
  n_jobs = 24
  producer = Producer(
    files,
    n_jobs=n_jobs,
    chunk_duration_s=3,
    target_sample_rate=48000,
    queue_size=n_jobs * 3,
  )

  worker = Worker(
    model_path,
    producer,
    batch_size=50,
    n_jobs=n_jobs,
    top_k=5,
    threshold=0.1,
    whitelist=None,
  )
  worker.start()

  producer.fill_queue()

  consumer = Consumer(producer, worker, init_w=1148)
  tensor = consumer.consume()

  print("before join")
  worker.join()
  print("after join")

  return tensor


if __name__ == "__main__":
  audio_path, duration = Path("example/soundscape.wav"), 2
  audio_path, duration = Path("test-dataset/test_dataset_1x1440min/0.wav"), 1440
  audio_path, duration = Path("test-dataset/test_dataset_1x60min/0.wav"), 60
  # 60 x 4: Finished analysis in 51.26 seconds.

  n_files = 4
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
  tst = analyze(
    paths,
    model_path=model_path,
  )
  end = perf_counter()
  print(f"Finished analysis in {end - start:.2f} seconds.")
  print(f"Finished analysis in {(end - start) / n_files:.2f} seconds per file.")
  print(f"Finished analysis in {(end - start) / n_files / duration * 60:.2f} s/h.")
  print(f"Finished analysis in {(end - start) / n_files / duration * 1000:.2f} ms/min.")
  print(tst.get_at(0, 2))
