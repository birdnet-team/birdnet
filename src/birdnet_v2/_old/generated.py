# birdnet_batch_inference.py — Shared‑Memory Ring‑Buffer edition
"""High‑throughput BirdNET inference pipeline.

Features
========
* **Raw audio** windows go straight into the model (no mel front‑end here).
* **Shared‑memory ring buffer** eliminates Pickle/pipe copies: only the
  `(slot_id, n_frames)` metadata travels through the queue.
* Works on **Linux/macOS** (fork+copy‑on‑write) *and* Windows (spawn) via
  `multiprocessing.shared_memory` (Python ≥ 3.8).
* Scales to 100 k+ files, dozens of CPU cores, while RAM stays predictable.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import sys
from multiprocessing import shared_memory
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from tensorflow.lite.python import interpreter as tflite  # type: ignore

# ---------------------------------------------------------------------------
# Audio & batching parameters
# ---------------------------------------------------------------------------
WIN_SEC = 3.0  # analysis window
HOP_SEC = 1.0  # hop size
SR = 48_000  # sample rate
WIN_SAMP = int(WIN_SEC * SR)  # ≈ 96 000 samples
BATCH_FRAMES = 512  # model batch
RING_SLOTS = 8  # 2 × worker‑count recommended
EMPTY_ID = 0xFFFF

# ---------------------------------------------------------------------------
# Helper – file‑iterator (yields single frames)
# ---------------------------------------------------------------------------
Frame = Tuple[int, int, np.ndarray]  # (file_idx, win_idx, frame)


def window_iterator(files: Sequence[str]) -> Iterable[Frame]:
  """Yield one raw‑audio window at a time for *all* files."""
  for f_idx, path in enumerate(files):
    with sf.SoundFile(path) as f:
      if f.samplerate != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {f.samplerate}")
      hop_samp = int(HOP_SEC * SR)
      win_samp = WIN_SAMP
      total_frames = int(math.floor((len(f) - win_samp) / hop_samp)) + 1
      for w_idx in range(total_frames):
        start = w_idx * hop_samp
        f.seek(start)
        frame = f.read(frames=win_samp, dtype="float32", always_2d=False)
        if frame.ndim == 2:
          frame = frame.mean(axis=1)
        yield f_idx, w_idx, frame


# ---------------------------------------------------------------------------
# SpeciesTensor – result container (unchanged)
# ---------------------------------------------------------------------------
class SpeciesTensor:
  def __init__(self, n_files: int, init_W: int, k: int, species_map: Sequence[str]):
    self.k = k
    self.species_map = species_map
    self._alloc(n_files, init_W)
    self.next_free = np.zeros(n_files, dtype=np.int32)

  def _alloc(self, F: int, W: int):
    self.ids = np.full((F, W, self.k), EMPTY_ID, np.uint16)
    self.probs = np.zeros((F, W, self.k), np.float16)
    self.mask = np.ones((F, W, self.k), bool)

  def ensure_capacity(self, w: int):
    if w < self.ids.shape[1]:
      return
    new_W = w + 1
    self.ids = np.resize(self.ids, (self.ids.shape[0], new_W, self.k))
    self.probs = np.resize(self.probs, (self.probs.shape[0], new_W, self.k))
    self.mask = np.resize(self.mask, (self.mask.shape[0], new_W, self.k))

  def write_scatter(self, f_ids, w_ids, idx, prob, msk):
    for j in range(len(f_ids)):
      f = f_ids[j]
      w = w_ids[j]
      self.ensure_capacity(w)
      self.ids[f, w] = idx[j]
      self.probs[f, w] = prob[j]
      self.mask[f, w] = msk[j]


# ---------------------------------------------------------------------------
# Worker callable – persistent TFLite interpreter + SHM access
# ---------------------------------------------------------------------------
class Worker:
  def __init__(
    self,
    *,
    model_path: Path,
    top_k: int,
    thresh: float,
    valid: np.ndarray,
    ring_name: str,
    n_slots: int,
  ):
    self.k = top_k
    self.thresh = thresh
    self.valid = valid
    # Interpreter
    self.interp = tflite.Interpreter(str(model_path), num_threads=1)
    self.interp.allocate_tensors()
    self.in_idx = self.interp.get_input_details()[0]["index"]
    self.out_idx = self.interp.get_output_details()[0]["index"]
    self.cached_shape: Optional[Tuple[int, int]] = None
    # map shared buffers
    # attatch to existing shared memory buffers
    shm_audio = shared_memory.SharedMemory(name=f"{ring_name}_audio", create=False)
    shm_file = shared_memory.SharedMemory(name=f"{ring_name}_file", create=False)
    shm_win = shared_memory.SharedMemory(name=f"{ring_name}_win", create=False)
    self.audio = np.ndarray(
      (n_slots, BATCH_FRAMES, WIN_SAMP), np.float32, shm_audio.buf
    )
    self.file = np.ndarray((n_slots, BATCH_FRAMES), np.uint32, shm_file.buf)
    self.win = np.ndarray((n_slots, BATCH_FRAMES), np.uint32, shm_win.buf)

  # ------------------------------------------------------------
  def _infer(self, batch: np.ndarray):
    if batch.shape != self.cached_shape:
      self.interp.resize_tensor_input(self.in_idx, batch.shape, strict=True)
      self.interp.allocate_tensors()
      self.cached_shape = batch.shape
    self.interp.set_tensor(self.in_idx, batch)
    self.interp.invoke()
    return self.interp.get_tensor(self.out_idx)

  # ------------------------------------------------------------
  def __call__(
    self,
    job_q: mp.Queue,
    out_q: mp.Queue,
    sem_free: mp.Semaphore,
    sem_fill: mp.Semaphore,
  ):
    while True:
      job = job_q.get()
      if job is None:
        break
      slot, n = job
      sem_fill.acquire()
      pcm_view = self.audio[slot, :n]
      f_ids = self.file[slot, :n]
      w_ids = self.win[slot, :n]
      pred = self._infer(pcm_view)
      idx = np.argpartition(pred, -self.k, axis=1)[:, -self.k :]
      prob = np.take_along_axis(pred, idx, axis=1)
      order = np.argsort(-prob, axis=1)
      row = np.arange(n)[:, None]
      idx = idx[row, order]
      prob = prob[row, order]
      keep = (prob >= self.thresh) & self.valid[idx]
      msk = ~keep
      idx[msk] = EMPTY_ID
      prob[msk] = 0.0
      out_q.put(
        (
          f_ids,
          w_ids,
          idx.astype(np.uint16, copy=False),
          prob.astype(np.float16, copy=False),
          msk,
        )
      )
      sem_free.release()



def analyze(
  files: List[str],
  *,
  model_path: Path,
  chunk_sec: float = 18.0,
  batch_frames: int = BATCH_FRAMES,
  J: int = os.cpu_count() or 1,
  top_k: int = 5,
  threshold: float = 0.1,
  whitelist: Optional[Sequence[int]] = None,
  species_map: Optional[Sequence[str]] = None,
) -> SpeciesTensor:
  species_map = species_map or [f"sp_{i}" for i in range(6522)]
  n_files = len(files)

  # ---------- Whitelist mask ---------------------------------
  valid = np.zeros(len(species_map), bool)
  if whitelist:
    valid[list(whitelist)] = True
  else:
    valid[:] = True
  valid.setflags(write=False)

  # ---------- Shared memory ring -----------------------------
  ring_name = "bnet_ring"

  shm_audio = create_ring(
    f"{ring_name}_audio", RING_SLOTS * batch_frames * WIN_SAMP * 4
  )
  shm_file = create_ring(size=RING_SLOTS * batch_frames * 4, name=f"{ring_name}_file")
  shm_win = create_ring(size=RING_SLOTS * batch_frames * 4, name=f"{ring_name}_win")
  ring_audio = np.ndarray(
    (RING_SLOTS, batch_frames, WIN_SAMP), np.float32, shm_audio.buf
  )
  ring_file = np.ndarray((RING_SLOTS, batch_frames), np.uint32, shm_file.buf)
  ring_win = np.ndarray((RING_SLOTS, batch_frames), np.uint32, shm_win.buf)

  sem_free = mp.Semaphore(RING_SLOTS)  # counts free slots
  sem_fill = mp.Semaphore(0)  # counts filled slots
  write_ptr = mp.Value("I", 0)

  job_q: mp.Queue = mp.Queue()
  out_q: mp.Queue = mp.Queue(maxsize=4 * J)

  # ---------- Worker pool ------------------------------------
  workers: List[mp.Process] = []
  for _ in range(J):
    w = Worker(
      model_path=model_path,
      top_k=top_k,
      thresh=threshold,
      valid=valid,
      ring_name=ring_name,
      n_slots=RING_SLOTS,
    )
    p = mp.Process(target=w, args=(job_q, out_q, sem_free, sem_fill), daemon=True)
    p.start()
    workers.append(p)

  # ---------- Producer ---------------------------------------
  def producer():
    buf_pcm, buf_f, buf_w = [], [], []
    for f_idx, w_idx, frame in window_iterator(files):
      buf_pcm.append(frame)
      buf_f.append(f_idx)
      buf_w.append(w_idx)
      if len(buf_pcm) == batch_frames:
        _flush_batch(buf_pcm, buf_f, buf_w)
    if buf_pcm:
      _flush_batch(buf_pcm, buf_f, buf_w)
    # send poison pills
    for _ in workers:
      job_q.put(None)

  def _flush_batch(pcm_list, f_list, w_list):
    sem_free.acquire()
    with write_ptr.get_lock():
      slot = write_ptr.value % RING_SLOTS
      write_ptr.value += 1
    n = len(pcm_list)
    ring_audio[slot, :n] = np.stack(pcm_list, 0)
    ring_file[slot, :n] = np.asarray(f_list, np.uint32)
    ring_win[slot, :n] = np.asarray(w_list, np.uint32)
    job_q.put((slot, n))
    sem_fill.release()
    pcm_list.clear()
    f_list.clear()
    w_list.clear()

  prod = mp.Process(target=producer, daemon=True)
  prod.start()

  # ---------- Writer / result gathering ----------------------
  tensor = SpeciesTensor(n_files, int(chunk_sec / HOP_SEC) + 1, top_k, species_map)
  live = J
  while live:
    try:
      out = out_q.get(timeout=1.0)
    except queue.Empty:
      live = sum(p.is_alive() for p in workers)
      continue
    f_ids, w_ids, idx, prob, msk = out
    tensor.write_scatter(f_ids, w_ids, idx, prob, msk)

  # drain leftovers
  while not out_q.empty():
    f_ids, w_ids, idx, prob, msk = out
  return tensor

import atexit
from multiprocessing import shared_memory

RING_NAMES = ["bnet_ring_audio", "bnet_ring_file", "bnet_ring_win"]


def create_ring(name: str, size: int):
  shm = shared_memory.SharedMemory(create=True, name=name, size=size)
  atexit.register(_cleanup_shm, name)  # wird auch bei Ctrl-C ausgeführt
  return shm


def _cleanup_shm(name: str) -> None:
  try:
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()
    print(f"unlinked {name}")
  except FileNotFoundError:
    pass


if __name__ == "__main__":
  audio_path = Path("test-dataset/test_dataset_1x1440min/0.wav")
  audio_path = Path("example/soundscape.wav")
  audio_path = Path("test-dataset/test_dataset_1x60min/0.wav")
  duration = 60

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
  tst = analyze(
    paths,
    model_path=model_path,
    J=1,
  )
  end = perf_counter()
  print(f"Finished analysis in {end - start:.2f} seconds.")
  print(f"Finished analysis in {(end - start) / n_files:.2f} seconds per file.")
  print(f"Finished analysis in {(end - start) / n_files:.2f} s/h.")
  print(f"Finished analysis in {(end - start) / n_files / duration * 1000:.2f} ms/min.")
