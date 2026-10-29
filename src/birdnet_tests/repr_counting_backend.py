"""A fake backend that counts NumPy array renderings inside its worker process.

The worker's per-batch debug lines carry the batch's segment indices, a NumPy
array. Rendering one costs tens of microseconds, so whether the pipeline
renders it at the default INFO level is a measurable property of the worker
process -- and only of that process, which is why the count is taken there
and not in the test.
"""

from __future__ import annotations

import atexit
import os
from pathlib import Path

import numpy as np

from birdnet_tests.fake_acoustic_backend import FakeAcousticBackend

COUNT_DIR_ENV_VAR = "BIRDNET_TEST_REPR_COUNT_DIR"


def _arrayprint():  # noqa: ANN202
  try:
    import numpy._core.arrayprint as arrayprint
  except ImportError:  # numpy 1.x
    import numpy.core.arrayprint as arrayprint
  return arrayprint


class ReprCountingBackend(FakeAcousticBackend):
  """Wraps NumPy's array renderer in `load()` and reports the count per process.

  `_array2string` is the one function behind `str()`, `repr()` and `%s` or
  f-string formatting of an ndarray. The output directory comes from the
  environment because the backend is built inside the spawned worker, which
  inherits the parent's environment but none of its objects. The count is
  written on every `predict()`, so the file is current after each batch; the
  write at interpreter exit covers a clean exit only.
  """

  def load(self) -> None:
    arrayprint = _arrayprint()
    self._count = 0
    self._count_file = Path(os.environ[COUNT_DIR_ENV_VAR]) / f"reprs-{os.getpid()}.txt"
    original = arrayprint._array2string

    def counting(*args: object, **kwargs: object) -> object:
      self._count += 1
      return original(*args, **kwargs)

    arrayprint._array2string = counting
    self._write()
    atexit.register(self._write)

  def _write(self) -> None:
    self._count_file.write_text(str(self._count), encoding="utf-8")

  def predict(self, batch: np.ndarray) -> np.ndarray:
    self._write()
    return super().predict(batch)


def read_counts(directory: Path) -> dict[int, int]:
  """Worker pid -> number of array renderings that worker reported."""
  return {
    int(path.stem.removeprefix("reprs-")): int(path.read_text(encoding="utf-8"))
    for path in directory.glob("reprs-*.txt")
  }
