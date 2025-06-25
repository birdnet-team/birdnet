import atexit
import contextlib
import logging
import multiprocessing
import multiprocessing as mp
import os
from collections.abc import Generator, Iterable
from itertools import count, islice
from logging import getLogger
from logging.handlers import QueueHandler
from multiprocessing import Queue, shared_memory
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from time import sleep
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import requests
import soundfile as sf
from numpy.typing import DTypeLike
from ordered_set import OrderedSet
from scipy.signal import butter, lfilter, resample
from tqdm import tqdm

import birdnet_v2.logging_utils as bn_logging
from birdnet.types import Species, TimeInterval
from birdnet.utils import (
  bandpass_signal,
  fillup_with_silence,
  get_chunks_with_overlap,
  itertools_batched,
  resample_array,
)
from birdnet_v2.globals import DONE_FLAG, READABLE_FLAG, WRITABLE_FLAG
from birdnet_v2.helper import (
  RingField,
  max_value_for_uint_dtype,
  uint_dtype_for,
)
from birdnet_v2.inference.producer import (
  Producer,
  load_audio_in_chunks_with_overlap,
)


@contextlib.contextmanager
def shm_ring_from_name(name: str, size: int):
  shm = SharedMemory(create=True, name=name, size=size)
  try:
    yield shm
  finally:
    shm.close()
    shm.unlink()  # wird sogar bei CTRL-C im finally ausgeführt
    print(f"Shared memory {name} cleaned up.")


def test_producing():
  import faulthandler
  import sys

  faulthandler.enable(file=sys.stderr, all_threads=True)

  audio_path = Path("test-dataset/test_dataset_1x1440min/0.wav")
  audio_path = Path("example/soundscape.wav")
  n_files = 2
  files = [audio_path] * n_files

  batch_size = 3
  n_slots = 8
  n_workers = 4
  prod_queue = Queue()
  sem_free = mp.Semaphore(n_slots)
  sem_fill = mp.Semaphore()
  chunk_duration_s = 3
  overlap_duration_s = 0
  target_sample_rate = 48000
  chunk_duration_samples = int(chunk_duration_s * target_sample_rate)

  with (
    shm_ring_from_name(
      "bnet_ring_file_indices",
      n_slots * batch_size * np.dtype(np.uint32).itemsize,
    ) as shm_file_indices,
    shm_ring_from_name(
      "bnet_ring_chunk_indices", n_slots * batch_size * np.dtype(np.uint32).itemsize
    ) as shm_chunk_indices,
    shm_ring_from_name(
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
        n_workers,
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

    ring_file_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, shm_file_indices.buf
    )
    ring_chunk_indices = np.ndarray(
      (n_slots, batch_size), np.uint32, shm_chunk_indices.buf
    )
    ring_audio_samples = np.ndarray(
      (n_slots, batch_size, chunk_duration_samples),
      np.float32,
      shm_audio_samples.buf,
    )

    res = []
    while True:
      job = prod_queue.get()
      if job is None:
        break
      slot, n = job
      sem_fill.acquire()
      file_indices = ring_file_indices[slot, :n]
      chunk_indices = ring_chunk_indices[slot, :n]
      audio_samples = ring_audio_samples[slot, :n]
      res.append(list(chunk_indices))
      sem_free.release()
  print("Producer finished processing.")
  print(res)


def test_chunking():
  from time import perf_counter

  t1 = perf_counter()
  res = list(
    load_audio_in_chunks_with_overlap(Path("test-dataset/test_dataset_1x1440min/0.wav"))
  )
  print(f"Loaded {len(res)} chunks in {perf_counter() - t1:.2f} seconds")


if __name__ == "__main__":
  test_producing()
