from contextlib import contextmanager, suppress
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from birdnet.acoustic_models.inference_pipeline.logs import get_logger_from_session


@dataclass(slots=True, frozen=True)
class RingField:
  name: str
  dtype: np.dtype
  shape: tuple[int, ...]

  # ----------------------------------------
  @property
  def nbytes(self) -> int:
    return int(np.prod(self.shape)) * self.dtype.itemsize

  def attach_shared_memory(self) -> shared_memory.SharedMemory:
    """
    Attaches to an existing shared memory segment with the specified name.
    """
    return shared_memory.SharedMemory(name=self.name, create=False)

  def cleanup(self, session_id: str) -> None:
    try:
      shm = self.attach_shared_memory()
    except FileNotFoundError:
      return
    else:
      logger = get_logger_from_session(session_id, __name__)
      logger.debug(f"Cleaning up shared memory {self.name}.")
      shm.close()
      with suppress(FileNotFoundError):
        shm.unlink()
      logger.debug(f"Shared memory {self.name} cleaned up.")

  def get_array(self, shm: shared_memory.SharedMemory) -> np.ndarray:
    view = np.ndarray(self.shape, self.dtype, buffer=shm.buf)
    return view

  def attach_and_get_array(self) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    shm = self.attach_shared_memory()
    view = self.get_array(shm)
    return shm, view


@contextmanager  # type: ignore
def create_shm_ring(session_id: str, ring: RingField) -> shared_memory.SharedMemory:  # type: ignore
  shm = shared_memory.SharedMemory(name=ring.name, create=True, size=ring.nbytes)
  try:
    yield shm  # type: ignore
  finally:
    shm.close()
    with suppress(FileNotFoundError):
      shm.unlink()
    logger = get_logger_from_session(session_id, __name__)
    logger.debug(f"Shared memory {ring.name} cleaned up.")
