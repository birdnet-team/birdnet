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
    *,
    model_path: Path,
) -> SpeciesTensor:
    producer = Producer(
        files,
        chunk_duration_s=3,
        target_sample_rate=48000,
        queue_size=16*600,
    )

    worker = Worker(
        model_path,
        producer,
        batch_size=2,
        n_jobs=16,
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
    audio_path = Path("test-dataset/test_dataset_1x1440min/0.wav")
    audio_path = Path("example/soundscape.wav")
    audio_path = Path("test-dataset/test_dataset_1x60min/0.wav")
    duration = 60

    n_files = 16
    paths = [audio_path] * n_files
    model_path = Path(
        "/home/stefan/.local/share/birdnet/models/v2.4/TFLite/audio-model.tflite")
    model_path = Path(
        "src/birdnet_legacy/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
    import timeit
    from time import perf_counter
    start = perf_counter()
    print(f"Started analysis... {time.strftime('%H:%M:%S')}")
    tst = analyze(
        paths,
        model_path=model_path,
    )
    end = perf_counter()
    print(f"Finished analysis in {end - start:.2f} seconds.")
    print(
        f"Finished analysis in {(end - start)/n_files:.2f} seconds per file.")
    print(f"Finished analysis in {(end - start)/n_files:.2f} s/h.")
    print(
        f"Finished analysis in {(end - start)/n_files/duration*1000:.2f} ms/min.")
    print(tst.get_at(0, 2))
