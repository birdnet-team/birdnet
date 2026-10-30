"""The memory budget behind the default worker count.

Every worker is its own process that imports its runtime and loads its own copy
of the model (the start method is ``spawn`` by default on every platform), so
peak memory grows with the worker count while the physical core count, the old
default, says nothing about what fits. The estimate a session plans with is

  peak ~ parent + n_producers * producer
         + n_workers * (worker + (1 + prefetch_ratio) * slot)
  slot = batch_size * segment_samples * 4 bytes          (the audio ring)
  producer = producer_base + 1.5 * slot                  (the batch it builds)
  worker = runtime + model_factor * model_bytes + batch_size * per_segment

with ``runtime``, ``model_factor`` and ``per_segment`` per backend from
``_WORKER_COSTS``. ``n_workers=None`` becomes the largest count whose peak fits
90 % of the memory available to this process after the reserve for the parent
and the producers, capped at the physical core count and never below one; an
explicit ``n_workers`` is taken as given.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from birdnet.core.backends import (
  TF_BACKEND_LIB_ARG,
  OnnxBackend,
  PBBackend,
  TFBackend,
  TorchBackend,
  VersionedBackendProtocol,
)
from birdnet.globals import LIBRARY_LITERT
from birdnet.utils.logging_utils import get_logger_for_package

_LOGGER = get_logger_for_package(__name__)

_MIB = 2**20
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_SELF_CGROUP = Path("/proc/self/cgroup")
# cgroup v1 reports "no limit" as a number near 2**63.
_CGROUP_V1_NO_LIMIT = 2**60
# Kept back for the parent process and the performance tracker. Peak working set
# in the setup below: the parent grows 1-2 MiB during a run (57 MiB with
# show_stats="benchmark") and the tracker takes 46 MiB; the rest is for the queue
# readers' buffers and the first growth steps of the result tensor, which can
# outgrow this on a large corpus.
_PARENT_RESERVE_BYTES = 128 * _MIB
# Kept back per producer, on top of the batch it builds. Peak working set of one
# producer at batch_size=1, Windows 11, spawn, Python 3.13, numpy 2.5, scipy
# 1.18, 14 logical cores, n_workers=1, 2.4 tf/pb and 3.0 tf/onnx alike: 45 MiB
# reading at the model's rate, 111 MiB when it resamples (scipy); its commit is
# 31 and 86 MiB (2.4 tf, 3.0 onnx). Not measured: Linux, macOS, array inputs.
_PRODUCER_RESERVE_BYTES = 128 * _MIB
# Ring slots a producer holds privately while it builds a batch: the stacked
# batch and the resample temporaries. Measured 1.2 slots of growth from batch
# size 1 to 256 in the setup above, resampling or not.
_PRODUCER_SLOTS = 1.5
# The estimates below are 3-5 % under the largest measured worker, and a worker
# with more intra-op threads (fewer workers share the cores) needs more.
_PLANNING_FRACTION = 0.9


@dataclass(frozen=True)
class WorkerCost:
  runtime_bytes: int
  model_factor: float
  per_segment_bytes: int

  def estimate(self, model_bytes: int, batch_size: int) -> int:
    return int(
      self.runtime_bytes
      + self.model_factor * model_bytes
      + batch_size * self.per_segment_bytes
    )


# Max RssAnon of one worker, Linux x86_64, spawn, Python 3.13, TensorFlow 2.21,
# onnxruntime 1.24, fp32 models, 12 cores shared by the workers: tflite 236 MiB
# (49 MiB model), pb 2.4 ~700 MiB (56 MiB model) and pb 3.0 ~1800 MiB (520 MiB
# model), onnx 3.0 730 MiB (516 MiB model); batch 8 added 12-18 MiB per
# segment. Not measured: litert, pt, fp16 and int8 files (which the runtimes
# may widen to fp32), the 3.0 tflite file, Perch V2 and GPU devices. "other"
# is any backend outside this module's classes, such as the test stub.
_WORKER_COSTS: dict[str, WorkerCost] = {
  "tflite": WorkerCost(180 * _MIB, 1.2, 15 * _MIB),
  "litert": WorkerCost(60 * _MIB, 1.2, 15 * _MIB),
  "pb": WorkerCost(550 * _MIB, 2.4, 18 * _MIB),
  "onnx": WorkerCost(150 * _MIB, 1.1, 12 * _MIB),
  "pt": WorkerCost(300 * _MIB, 1.2, 15 * _MIB),
  "other": WorkerCost(30 * _MIB, 1.2, 15 * _MIB),
}


def _read_int(path: Path) -> int | None:
  try:
    text = path.read_text().strip()
  except (OSError, ValueError):
    return None
  if text == "max":
    return None
  try:
    return int(text)
  except ValueError:
    return None


def _read_stat(path: Path, key: str) -> int:
  try:
    for line in path.read_text().splitlines():
      name, _, value = line.partition(" ")
      if name == key:
        return int(value)
  except (OSError, ValueError):
    pass
  return 0


def cgroup_memory_available(
  cgroup_root: Path = _CGROUP_ROOT, proc_self_cgroup: Path = _PROC_SELF_CGROUP
) -> int | None:
  """Bytes left under the tightest cgroup memory limit, ``None`` without one.

  Page cache counts as free, since the kernel reclaims it before it kills;
  shared memory, which it cannot reclaim, does not. A container whose
  ``/proc/self/cgroup`` names a path the mounted hierarchy does not have (the
  mount is the container's own cgroup) is read at the mount root.
  """
  try:
    lines = proc_self_cgroup.read_text().splitlines()
  except (OSError, ValueError):
    return None
  available: int | None = None

  def narrow(limit: int, used: int) -> None:
    nonlocal available
    left = max(0, limit - max(0, used))
    available = left if available is None else min(available, left)

  for line in lines:
    _, _, rest = line.partition(":")
    controllers, _, path = rest.partition(":")
    rel = path.strip("/")
    if controllers == "":
      base = cgroup_root / rel if rel else cgroup_root
      # A path the mount does not have (a container's own cgroup mounted as
      # the root) is harmless: the walk ends at the mount root either way.
      for d in (base, *base.parents):
        limit = _read_int(d / "memory.max")
        if limit is not None:
          stat = d / "memory.stat"
          cache = _read_stat(stat, "file") - _read_stat(stat, "shmem")
          narrow(limit, (_read_int(d / "memory.current") or 0) - cache)
        if d == cgroup_root:
          break
    elif "memory" in controllers.split(","):
      d = cgroup_root / "memory" / rel if rel else cgroup_root / "memory"
      if not d.is_dir():
        d = cgroup_root / "memory"
      stat = d / "memory.stat"
      limits = [
        _read_int(d / "memory.limit_in_bytes"),
        _read_stat(stat, "hierarchical_memory_limit") or None,
      ]
      real = [lim for lim in limits if lim is not None and lim < _CGROUP_V1_NO_LIMIT]
      if real:
        cache = _read_stat(stat, "total_cache") - _read_stat(stat, "total_shmem")
        narrow(min(real), (_read_int(d / "memory.usage_in_bytes") or 0) - cache)
  return available


def host_memory_available() -> int:
  return int(psutil.virtual_memory().available)


def available_memory() -> tuple[int, str]:
  """What this process may still allocate, and which view set it."""
  host = host_memory_available()
  cgroup = cgroup_memory_available()
  if cgroup is not None and cgroup < host:
    return cgroup, "cgroup limit"
  return host, "host"


def available_memory_bytes() -> int:
  return available_memory()[0]


def physical_cores() -> int:
  return psutil.cpu_count(logical=False) or 1


def model_bytes(model_path: Path) -> int:
  """Size of the model on disk: one file, or every file of a SavedModel directory."""
  try:
    if model_path.is_dir():
      return sum(p.stat().st_size for p in model_path.rglob("*") if p.is_file())
    return model_path.stat().st_size
  except OSError:
    return 0


def cost_key(
  backend_type: type[VersionedBackendProtocol], backend_kwargs: dict[str, Any]
) -> str:
  if issubclass(backend_type, PBBackend):
    return "pb"
  if issubclass(backend_type, TFBackend):
    litert = backend_kwargs.get(TF_BACKEND_LIB_ARG) == LIBRARY_LITERT
    return "litert" if litert else "tflite"
  if issubclass(backend_type, OnnxBackend):
    return "onnx"
  if issubclass(backend_type, TorchBackend):
    return "pt"
  return "other"


def estimate_worker_bytes(
  backend_type: type[VersionedBackendProtocol],
  backend_kwargs: dict[str, Any],
  model_bytes: int,
  batch_size: int,
) -> int:
  """Peak private memory of one worker: runtime import, model, one batch."""
  cost = _WORKER_COSTS[cost_key(backend_type, backend_kwargs)]
  return cost.estimate(model_bytes, batch_size)


def per_worker_bytes(worker_bytes: int, slot_bytes: int, prefetch_ratio: int) -> int:
  """A worker and the ring slots it brings with it."""
  return worker_bytes + (1 + prefetch_ratio) * slot_bytes


def reserve_bytes(n_producers: int, slot_bytes: int) -> int:
  """Kept back from the workers' budget: the parent and every producer."""
  producer = _PRODUCER_RESERVE_BYTES + int(_PRODUCER_SLOTS * slot_bytes)
  return _PARENT_RESERVE_BYTES + n_producers * producer


def fit_n_workers(
  *,
  n_cores: int,
  available_bytes: int,
  n_producers: int,
  worker_bytes: int,
  slot_bytes: int,
  prefetch_ratio: int,
) -> int:
  """The largest worker count up to ``n_cores`` whose estimated peak fits."""
  per_worker = per_worker_bytes(worker_bytes, slot_bytes, prefetch_ratio)
  reserve = reserve_bytes(n_producers, slot_bytes)
  budget = int((available_bytes - reserve) * _PLANNING_FRACTION)
  fitting = budget // per_worker if per_worker > 0 else n_cores
  return max(1, min(n_cores, int(fitting)))


def default_n_workers(
  *,
  backend_type: type[VersionedBackendProtocol],
  backend_kwargs: dict[str, Any],
  model_path: Path,
  n_producers: int,
  batch_size: int,
  prefetch_ratio: int,
  segment_samples: int,
) -> int:
  """The worker count for ``n_workers=None``, logged with its reasons at INFO."""
  n_cores = physical_cores()
  available, source = available_memory()
  model = model_bytes(model_path)
  worker = estimate_worker_bytes(backend_type, backend_kwargs, model, batch_size)
  slot = batch_size * segment_samples * 4
  n_workers = fit_n_workers(
    n_cores=n_cores,
    available_bytes=available,
    n_producers=n_producers,
    worker_bytes=worker,
    slot_bytes=slot,
    prefetch_ratio=prefetch_ratio,
  )
  per_worker = per_worker_bytes(worker, slot, prefetch_ratio)
  decision = (
    "one per physical core"
    if n_workers == n_cores
    else f"{n_cores} would need ~{n_cores * per_worker // _MIB} MiB"
  )
  producers = "1 producer" if n_producers == 1 else f"{n_producers} producers"
  _LOGGER.info(
    f"n_workers defaulted to {n_workers} ({decision}): {available // _MIB} MiB "
    f"available ({source}); {_PLANNING_FRACTION * 100:.0f} % of what is left "
    f"after {reserve_bytes(n_producers, slot) // _MIB} MiB for the parent and "
    f"{producers} goes to workers of ~{per_worker // _MIB} MiB each "
    f"({cost_key(backend_type, backend_kwargs)} backend, {model // _MIB} MiB "
    f"model, batch size {batch_size}, "
    f"{1 + prefetch_ratio} ring slots of {slot / _MIB:.1f} MiB)."
  )
  return n_workers
