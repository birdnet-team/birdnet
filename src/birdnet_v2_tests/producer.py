import contextlib
import multiprocessing
import multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np

from birdnet_v2.acoustic_models.inference.producer import (
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
