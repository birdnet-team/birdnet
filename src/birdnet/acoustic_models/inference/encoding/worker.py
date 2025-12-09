from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.synchronize import Event, Lock, Semaphore

import numpy as np

from birdnet.acoustic_models.inference.worker import WorkerBase
from birdnet.backends import BackendLoader, BatchT
from birdnet.shm import RingField


class EmbeddingsWorker(WorkerBase):
  def __init__(
    self,
    session_id: str,
    backend_loader: BackendLoader,
    batch_size: int,
    n_slots: int,
    rf_file_indices: RingField,
    rf_segment_indices: RingField,
    rf_audio_samples: RingField,
    rf_batch_sizes: RingField,
    rf_flags: RingField,
    segment_duration_samples: int,
    out_q: Queue,
    wkr_ring_access_lock: Lock,
    sem_free: Semaphore,
    sem_fill: Semaphore,
    sem_active_workers: Semaphore | None,
    half_precision: bool,
    wkr_stats_queue: Queue | None,
    logging_queue: Queue,
    logging_level: int,
    device: str,
    cancel_event: Event,
    prd_all_done_event: Event,
    start_signal: Event,
    end_event: Event,
  ) -> None:
    super().__init__(
      session_id=session_id,
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
      half_precision=half_precision,
      wkr_stats_queue=wkr_stats_queue,
      logging_queue=logging_queue,
      logging_level=logging_level,
      device=device,
      cancel_event=cancel_event,
      all_producers_finished=prd_all_done_event,
      start_signal=start_signal,
      end_event=end_event,
    )

  def _get_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    infer_result: np.ndarray,
  ) -> tuple[np.ndarray, ...]:
    return (file_indices, segment_indices, infer_result)

  def _infer(self, batch: BatchT) -> BatchT:
    assert self._backend is not None
    return self._backend.encode(batch)
