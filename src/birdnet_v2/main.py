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
import soundfile as sf  # pip install soundfile

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)
from tensorflow.lite.python import interpreter as tflite

from birdnet_v2.consumer import Consumer, SpeciesTensor
from birdnet_v2.producer import (  # type: ignore
  Producer,
  load_audio_in_chunks_with_overlap,
)
from birdnet_v2.worker import Worker

# ----------------------------------------------------------------------------
WIN_SEC = 3.0  # window length in seconds
HOP_SEC = 1.0  # hop size (sec)
SR = 32_000  # sample rate expected by the model
WIN_SAMPLES = int(WIN_SEC * SR)
EMPTY_ID = 0xFFFF

FrameBatch = Tuple[int, int, np.ndarray]  # (file_idx, first_win_idx, batch[B, N])
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import as_strided

WIN_SEC = 3.0
HOP_SEC = 1.0

import numpy.typing as npt


def analyze(
  files: List[Path],
  *,
  model_path: Path,
) -> SpeciesTensor:
  producer = Producer(
    files,
    chunk_duration_s=3,
    target_sample_rate=48000,
    queue_size=4,
  )

  worker = Worker(
    model_path,
    producer,
    batch_size=4,
    n_jobs=1,
    top_k=5,
    threshold=0.1,
    whitelist=None,
  )
  worker.start()

  producer.fill_queue()

  consumer = Consumer(producer, worker, init_w=7)
  tensor = consumer.consume()

  worker.join()

  return tensor


if __name__ == "__main__":
  path = Path("test-dataset/test_dataset_1x1440min/0.wav")
  path = Path("example/soundscape.wav")
  tst = analyze(
    [path],
    model_path=Path(
      "/home/stefan/.local/share/birdnet/models/v2.4/TFLite/audio-model.tflite"
    ),
  )
  print(tst.get_at(0, 2))
