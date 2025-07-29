from __future__ import annotations

import multiprocessing as mp
import multiprocessing.synchronize
from multiprocessing import Queue
from multiprocessing.synchronize import Event, Semaphore

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic_models.inference.worker import WorkerBase
from birdnet.acoustic_models.inference.backends import InferenceBackendLoader
from birdnet.helper import RingField


class EmbeddingsWorker(WorkerBase):
  def __init__(
    self,
    backend_loader: InferenceBackendLoader,
    batch_size: int,
    n_slots: int,
    rf_file_indices: RingField,
    rf_segment_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    segment_duration_samples: int,
    out_q: Queue,
    wkr_ring_access_lock: multiprocessing.synchronize.Lock,
    sem_free: Semaphore,
    sem_fill: Semaphore,
    sem_active_workers: Semaphore | None,
    emb_dtype: DTypeLike,
    wkr_stats_queue: mp.Queue | None,
    logging_queue: mp.Queue,
    logging_level: int,
    device: str,
    cancel_event: Event,
    prd_all_done_event: Event,
  ):
    super().__init__(
      name=__name__,
      backend_loader=backend_loader,
      batch_size=batch_size,
      n_slots=n_slots,
      rf_file_indices=rf_file_indices,
      rf_segment_indices=rf_segment_indices,
      rf_audio_samples=rf_audio_samples,
      rf_batch_sizes=rf_batch_sizes,
      rf_flags=rf_flags,
      segment_duration_samples=segment_duration_samples,
      out_q=out_q,
      wkr_ring_access_lock=wkr_ring_access_lock,
      sem_free=sem_free,
      sem_fill=sem_fill,
      sem_active_workers=sem_active_workers,
      infer_dtype=emb_dtype,
      wkr_stats_queue=wkr_stats_queue,
      logging_queue=logging_queue,
      logging_level=logging_level,
      device=device,
      cancel_event=cancel_event,
      prd_all_done_event=prd_all_done_event,
    )

  def _get_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    infer_result: np.ndarray,
  ) -> tuple[np.ndarray, ...]:
    return (file_indices, segment_indices, infer_result)
