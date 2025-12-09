from __future__ import annotations

import multiprocessing.synchronize
from multiprocessing import Queue
from multiprocessing.synchronize import Event, Semaphore
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import DTypeLike

from birdnet.acoustic_models.inference.worker import WorkerBase
from birdnet.backends import BackendLoader, BatchT
from birdnet.helper import get_uint_dtype
from birdnet.shm import RingField
from birdnet.utils import flat_sigmoid_logaddexp_fast

if TYPE_CHECKING:
  pass


class PredictionWorker(WorkerBase):
  def __init__(
    self,
    session_id: str,
    backend_loader: BackendLoader,
    top_k: int,
    species_thresholds: np.ndarray,
    species_blacklist: np.ndarray,
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
    half_precision: bool,
    apply_sigmoid: bool,
    sigmoid_sensitivity: float | None,
    wkr_stats_queue: Queue | None,
    logging_queue: Queue,
    logging_level: int,
    device: str,
    cancel_event: Event,
    all_producers_finished: Event,
    start_signal: Event,
    end_event: Event,
  ) -> None:
    assert species_thresholds.shape[0] == 1
    assert species_blacklist.shape[0] == 1
    assert species_thresholds.shape[1] == species_blacklist.shape[1]
    assert species_thresholds.flags.aligned
    assert species_blacklist.flags.aligned

    self._top_k = top_k
    self._thresholds = species_thresholds
    self._blacklist = species_blacklist
    self._apply_sigmoid = apply_sigmoid
    self._sigmoid_sensitivity = None
    if apply_sigmoid:
      assert sigmoid_sensitivity is not None
      self._sigmoid_sensitivity = sigmoid_sensitivity
    self._batch_idx_cache = {}

    self._species_dtype: DTypeLike | None = None

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
      all_producers_finished=all_producers_finished,
      start_signal=start_signal,
      end_event=end_event,
    )

  def _infer(self, batch: BatchT) -> BatchT:
    assert self._backend is not None
    return self._backend.predict(batch)

  def _get_block(
    self,
    file_indices: np.ndarray,
    segment_indices: np.ndarray,
    infer_result: np.ndarray,
  ) -> tuple[np.ndarray, ...]:
    if self._species_dtype is None:
      n_species = infer_result.shape[1]
      self._species_dtype = get_uint_dtype(n_species - 1)

    if self._apply_sigmoid:
      assert self._sigmoid_sensitivity is not None
      infer_result = flat_sigmoid_logaddexp_fast(
        infer_result,
        sensitivity=-self._sigmoid_sensitivity,
      )

    invalid_mask = (infer_result < self._thresholds) | self._blacklist

    # set invalid scores to -inf for top-k selection
    shadow = np.where(invalid_mask, -np.inf, infer_result)

    # select top-k species (order is random!)
    top_k_species = np.argpartition(shadow, -self._top_k, axis=1)[
      :, -self._top_k :
    ].astype(self._species_dtype, copy=False)

    batch_idx = self._get_batch_idx(infer_result.shape[0])
    top_k_scores = infer_result[batch_idx, top_k_species]
    top_k_mask = invalid_mask[batch_idx, top_k_species]
    return (
      file_indices,
      segment_indices,
      top_k_species,
      top_k_scores,
      top_k_mask,
    )

  def _get_batch_idx(self, batch_size: int) -> np.ndarray:
    if batch_size not in self._batch_idx_cache:
      self._batch_idx_cache[batch_size] = np.arange(batch_size)[:, None]
    return self._batch_idx_cache[batch_size]
