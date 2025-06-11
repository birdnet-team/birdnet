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

from birdnet_v2.producer import Producer


class Worker:
    def __init__(
        self,
        model_path: Path,
        producer: Producer,
        batch_size: int = 2,
        n_jobs: Optional[int] = None,
        top_k: int = 5,
        threshold: float = 0.03,
        whitelist: Optional[Sequence[int]] = None,
    ) -> None:
        self.top_k = top_k

        self._queue = mp.Queue()

        species_map = [f"sp_{i}" for i in range(6522)]

        valid = np.zeros(len(species_map), bool)
        if whitelist:
            valid[list(whitelist)] = True
        else:
            valid[:] = True
        valid.setflags(write=False)

        if n_jobs is None:
            n_jobs = os.cpu_count()
            assert n_jobs is not None, "os.cpu_count() returned None"

        # workers
        self.workers = [
            mp.Process(
                target=ChildWorker(model_path, top_k, threshold, valid),
                args=(producer.queue, self._queue,
                      producer.reading_finished, batch_size),
                daemon=True,
            )
            for _ in range(n_jobs)
        ]

    @property
    def queue(self) -> mp.Queue:
        """
        Returns the queue used for processing audio files.
        """
        return self._queue

    def start(self) -> None:
        for p in self.workers:
            p.start()

    def join(self) -> None:
        for p in self.workers:
            p.join()


EMPTY_ID = 0  # 0xFFFF


# ----------------------------------------------------------------------------
class ChildWorker:
    def __init__(self, model_path: str, k: int, thresh: float, valid: np.ndarray):
        self.k = k
        self.thresh = thresh
        self.valid = valid
        self.interp = tflite.Interpreter(
            model_path=str(model_path), num_threads=1)
        self.interp.allocate_tensors()
        self.in_idx = self.interp.get_input_details()[0]["index"]
        self.out_idx = self.interp.get_output_details()[0]["index"]
        # cache last batch shape to avoid resize
        self.cached_shape: Optional[Tuple[int, int]] = None

    def _infer_topk(self, batch: np.ndarray):
        if self.cached_shape != batch.shape:
            self.interp.resize_tensor_input(
                self.in_idx, batch.shape, strict=True)
            self.interp.allocate_tensors()
            self.cached_shape = batch.shape
        self.interp.set_tensor(self.in_idx, batch)
        self.interp.invoke()
        pred = self.interp.get_tensor(self.out_idx)
        idx = np.argpartition(pred, -self.k, axis=1)[:, -self.k:]
        sorted_order = np.take_along_axis(
            pred, idx, axis=1).argsort(axis=1)[:, ::-1]
        idx_sorted = np.take_along_axis(idx, sorted_order, axis=1)
        prob = np.take_along_axis(pred, idx_sorted, axis=1)
        keep = (prob >= self.thresh) & self.valid[idx_sorted]
        msk = ~keep
        idx[msk] = EMPTY_ID
        prob[msk] = 0.0
        return idx.astype(np.uint16), prob.astype(np.float16), msk

    def __call__(self, job_q, out_q, stop, batch_size):
        buffer_audio = []
        buffer_meta = []

        while True:
            no_more_chunks_available = False
            chunks_available = not job_q.empty()
            if chunks_available:
                try:
                    file_idx, chunk_index, audio_chunk = job_q.get(timeout=0.1)
                except queue.Empty:
                    break

                buffer_audio.append(audio_chunk)
                buffer_meta.append((file_idx, chunk_index))
            else:
                queue_still_filling = not stop.is_set()
                if queue_still_filling:
                    time.sleep(0.1)
                    continue
                else:
                    no_more_chunks_available = True

            batch_is_full = len(buffer_audio) >= batch_size
            unprocessed_chunks_exist = len(buffer_audio) > 0

            clear_batch = batch_is_full

            if no_more_chunks_available and unprocessed_chunks_exist:
                clear_batch = True

            if clear_batch:
                batch = np.array(buffer_audio[:batch_size])

                species_indicies, species_probs, pred_msk = self._infer_topk(
                    batch)

                file_indices = np.array([f[0] for f in buffer_meta])
                chunk_indices = np.array([f[1] for f in buffer_meta])
                res = (file_indices, chunk_indices,
                       species_indicies, species_probs, pred_msk)
                out_q.put(res)

                leftover = np.array(buffer_audio[batch_size:])
                buffer_audio = [leftover] if leftover.size else []
                buffer_meta.clear()

                if no_more_chunks_available:
                    assert len(leftover) == 0
                    print("done", f"{job_q.empty()}")
                    print(f"outputqueue empty: {out_q.empty()}")
                    break

            if no_more_chunks_available:
                break

        print("left queue")
