#!/usr/bin/env python3
# ruff: noqa: T201
"""Benchmark birdnet inference on a fixed corpus: one leg, or a suite of legs.

Self-contained on purpose: this one file is copied onto whichever machine is
being measured and run inside the virtualenv of the version under test. It needs
only the stdlib plus numpy, scipy, soundfile and psutil, which every birdnet
version installs, and it touches only the public API (`birdnet.load`,
`load_perch_v2`, `model.predict`/`encode` and, where the version has them, the
session methods), so the same file measures 0.2.16 and the current main.

  python bench_birdnet.py env
  python bench_birdnet.py suite --suite quick --label main-f51ce46 \
      --seed example/soundscape.wav --work ./bench-work
  python bench_birdnet.py compare --results ./bench-work/results

benchmarks/README.md has the workflow and the rules that keep the numbers honest.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = 2
# Seconds one timed pass should take; the leg takes as many corpus files as that
# needs, so a Raspberry Pi and a GPU server both get a measurable pass.
PASS_SECONDS = 12.0
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg")
ENV_OF_INTEREST = (
  "BIRDNET_APP_DATA",
  "CUDA_VISIBLE_DEVICES",
  "OMP_NUM_THREADS",
  "MKL_NUM_THREADS",
  "OPENBLAS_NUM_THREADS",
  "TF_ENABLE_ONEDNN_OPTS",
  "TF_NUM_INTRAOP_THREADS",
  "TF_NUM_INTEROP_THREADS",
)


# --------------------------------------------------------------------------- env


def _cpu_name() -> str:
  system = platform.system()
  with contextlib.suppress(Exception):
    if system == "Linux":
      lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
      # x86 names the CPU under "model name"; ARM boards only under "Model".
      for field in ("model name", "Model", "Hardware"):
        for line in lines:
          if line.split(":")[0].strip() == field:
            return line.split(":", 1)[1].strip()
    if system == "Darwin":
      out = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
      )
      if out.stdout.strip():
        return out.stdout.strip()
    if system == "Windows":
      import winreg

      key = winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
      )
      return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
  return platform.processor() or platform.machine()


def _nvidia_smi(*query: str) -> list[str]:
  exe = shutil.which("nvidia-smi")
  if exe is None:
    return []
  with contextlib.suppress(Exception):
    out = subprocess.run(
      [exe, f"--query-gpu={','.join(query)}", "--format=csv,noheader,nounits"],
      capture_output=True,
      text=True,
      timeout=20,
      check=False,
    )
    if out.returncode == 0:
      return [line.strip() for line in out.stdout.splitlines() if line.strip()]
  return []


def _machine() -> dict[str, Any]:
  import psutil

  return {
    "host": platform.node(),
    "os": f"{platform.system()} {platform.release()}",
    "arch": platform.machine(),
    "cpu": _cpu_name(),
    "physical_cores": psutil.cpu_count(logical=False),
    "logical_cores": psutil.cpu_count(logical=True),
    "ram_GiB": round(psutil.virtual_memory().total / 2**30, 1),
    "gpus": _nvidia_smi("name", "memory.total", "driver_version"),
  }


def _birdnet_source() -> dict[str, Any]:
  """Where `birdnet` is imported from, and the commit when that is a checkout.

  The distribution version of an editable install is whatever it was when the
  install ran, so the commit is the only trustworthy identity of a work tree.
  """
  import importlib.util

  spec = importlib.util.find_spec("birdnet")
  if spec is None or spec.origin is None:
    return {}
  package = Path(spec.origin).resolve().parent
  info: dict[str, Any] = {"path": str(package)}
  git = shutil.which("git")
  if git is None:
    return info
  for parent in package.parents:
    if (parent / ".git").exists():
      with contextlib.suppress(Exception):
        head = subprocess.run(
          [git, "-C", str(parent), "rev-parse", "--short", "HEAD"],
          capture_output=True,
          text=True,
          timeout=20,
          check=False,
        ).stdout.strip()
        dirty = subprocess.run(
          [git, "-C", str(parent), "status", "--porcelain", "--untracked-files=no"],
          capture_output=True,
          text=True,
          timeout=20,
          check=False,
        ).stdout
        info["commit"] = head
        info["dirty"] = sorted(line[3:] for line in dirty.splitlines())
      break
  return info


def _software() -> dict[str, Any]:
  import importlib.metadata

  versions: dict[str, str] = {}
  for dist in (
    "birdnet",
    "tensorflow",
    "tensorflow-cpu",
    "ai-edge-litert",
    "torch",
    "onnxruntime",
    "onnxruntime-gpu",
    "numpy",
    "scipy",
    "soundfile",
    "psutil",
  ):
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
      versions[dist] = importlib.metadata.version(dist)
  with contextlib.suppress(Exception):
    import soundfile

    versions["libsndfile"] = soundfile.__libsndfile_version__
  return {
    "python": platform.python_version(),
    "executable": sys.executable,
    "versions": versions,
    "birdnet_source": _birdnet_source(),
    "env": {k: os.environ[k] for k in ENV_OF_INTEREST if k in os.environ},
  }


def cmd_env(args: argparse.Namespace) -> None:
  print(json.dumps({"machine": _machine(), "software": _software()}, indent=2))


# ------------------------------------------------------------------------ corpus


def _corpus_dir_name(fmt: str, sr: int, files: int, minutes: float) -> str:
  seconds = minutes * 60
  length = f"{minutes:g}min" if seconds >= 60 else f"{seconds:g}s"
  rate = f"{sr / 1000:g}k"
  # "mp3_48k", not "mp348k".
  return f"{fmt}{'_' if fmt[-1].isdigit() else ''}{rate}_{files}x{length}"


def build_corpus(
  seed: Path, out_root: Path, *, files: int, minutes: float, sr: int, fmt: str
) -> Path:
  """Tile `seed` into `files` identical recordings of `minutes` each.

  Deterministic from the seed, so every machine measures the same audio. The
  files are hard links to one written file where the file system allows it;
  birdnet de-duplicates inputs by path, so they still count as distinct inputs.
  """
  import numpy as np
  import soundfile as sf

  target = out_root / _corpus_dir_name(fmt, sr, files, minutes)
  manifest_path = target / "corpus.json"
  manifest = {
    "seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
    "files": files,
    "minutes": minutes,
    "sample_rate": sr,
    "format": fmt,
  }
  if manifest_path.is_file():
    with contextlib.suppress(Exception):
      if json.loads(manifest_path.read_text(encoding="utf-8")) == manifest:
        return target
    shutil.rmtree(target)
  target.mkdir(parents=True, exist_ok=True)

  data, seed_sr = sf.read(seed, dtype="float32")
  if data.ndim > 1:
    data = data.mean(axis=1)
  if seed_sr != sr:
    from scipy.signal import resample_poly

    divisor = math.gcd(sr, seed_sr)
    data = resample_poly(data, sr // divisor, seed_sr // divisor).astype(np.float32)
  n_samples = round(minutes * 60 * sr)
  tiled = np.tile(data, math.ceil(n_samples / len(data)))[:n_samples]

  width = len(str(files - 1))
  first = target / f"part_{0:0{width}d}.{fmt}"
  sf.write(first, tiled, sr, subtype="PCM_16" if fmt in ("wav", "flac") else None)
  source, n_copies = first, 0
  for i in range(1, files):
    link = target / f"part_{i:0{width}d}.{fmt}"
    try:
      os.link(source, link)
    except OSError:
      # No hard links here, or this file has all it can take (1023 on NTFS):
      # the copy takes the next ones.
      shutil.copyfile(first, link)
      source, n_copies = link, n_copies + 1
  if n_copies > files // 2:
    size = (1 + n_copies) * first.stat().st_size / 2**30
    print(
      f"no hard links on this file system: {target} holds {1 + n_copies} copies, "
      f"{size:.1f} GiB",
      file=sys.stderr,
    )
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  return target


def cmd_corpus(args: argparse.Namespace) -> None:
  target = build_corpus(
    Path(args.seed),
    Path(args.out),
    files=args.files,
    minutes=args.minutes,
    sr=args.sr,
    fmt=args.format,
  )
  print(target)


def _corpus_files(corpus: Path) -> list[str]:
  files = sorted(str(p) for p in corpus.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
  if not files:
    raise SystemExit(f"no audio files under {corpus}")
  return files


def _corpus_id(corpus: Path) -> str:
  manifest = corpus / "corpus.json"
  if not manifest.is_file():
    return corpus.name
  # Hashed as parsed: the line endings of the file and 10 written as 10.0 are
  # the same corpus.
  values = json.loads(manifest.read_text(encoding="utf-8"))
  canonical = {
    key: int(value) if isinstance(value, float) and value.is_integer() else value
    for key, value in values.items()
  }
  digest = hashlib.sha256(json.dumps(canonical, indent=2).encode()).hexdigest()[:8]
  return f"{corpus.name}#{digest}"


# -------------------------------------------------------------------------- legs


@dataclasses.dataclass(frozen=True)
class Leg:
  version: str = "3.0"  # "2.4", "3.0" or "perch"
  backend: str = "onnx"
  library: str | None = None  # tf backend only: "tflite" or "litert"
  precision: str = "fp32"
  device: str = "CPU"
  task: str = "predict"  # or "encode"
  workers: int | None = None  # None: the library's default
  producers: int = 1
  batch_size: int = 1
  prefetch_ratio: int | None = None  # None: the library's default
  overlap: float = 0.0
  speed: float = 1.0
  top_k: int = 5
  stats: bool = False
  # A custom classifier file and its label file: loaded with load_custom.
  custom_model: str | None = None
  custom_labels: str | None = None
  corpus: str = "wav48k"  # suite only: which corpus the leg reads

  @property
  def name(self) -> str:
    parts = [self.version, self.backend]
    if self.custom_model:
      model = Path(self.custom_model)
      digest = hashlib.sha256(model.read_bytes()).hexdigest()[:6]
      parts.append(f"custom_{model.parent.name}_{digest}")
    if self.library:
      parts.append(self.library)
    if self.precision != "fp32":
      parts.append(self.precision)
    parts.append(self.device.lower().replace(":", ""))
    parts.append(f"w{self.workers if self.workers is not None else 'auto'}")
    parts.append(f"b{self.batch_size}")
    parts.append(f"p{self.producers}")
    if self.prefetch_ratio is not None:
      parts.append(f"pf{self.prefetch_ratio}")
    if self.overlap:
      parts.append(f"ov{self.overlap:g}")
    if self.speed != 1.0:
      parts.append(f"sp{self.speed:g}")
    if self.top_k != 5:
      parts.append(f"k{self.top_k}")
    if self.task != "predict":
      parts.append(self.task)
    if self.stats:
      parts.append("stats")
    return "-".join(parts)

  def cli(self) -> list[str]:
    args = [
      "--version", self.version,
      "--backend", self.backend,
      "--precision", self.precision,
      "--device", self.device,
      "--task", self.task,
      "--producers", str(self.producers),
      "--batch-size", str(self.batch_size),
      "--overlap", str(self.overlap),
      "--speed", str(self.speed),
      "--top-k", str(self.top_k),
    ]  # fmt: skip
    if self.library:
      args += ["--library", self.library]
    if self.workers is not None:
      args += ["--workers", str(self.workers)]
    if self.prefetch_ratio is not None:
      args += ["--prefetch-ratio", str(self.prefetch_ratio)]
    if self.stats:
      args.append("--stats")
    if self.custom_model:
      args += ["--custom-model", self.custom_model]
    if self.custom_labels:
      args += ["--custom-labels", self.custom_labels]
    return args


def _leg_from_args(args: argparse.Namespace) -> Leg:
  return Leg(
    version=args.version,
    backend="pb" if args.version == "perch" else args.backend,
    library=args.library,
    precision=args.precision,
    device=args.device,
    task=args.task,
    workers=args.workers,
    producers=args.producers,
    batch_size=args.batch_size,
    prefetch_ratio=args.prefetch_ratio,
    overlap=args.overlap,
    speed=args.speed,
    top_k=args.top_k,
    stats=args.stats,
    custom_model=args.custom_model,
    custom_labels=args.custom_labels,
  )


def _load_model(leg: Leg) -> Any:  # noqa: ANN401
  import birdnet

  if leg.version == "perch":
    return birdnet.load_perch_v2(leg.device.split(":")[0])
  kwargs: dict[str, Any] = {}
  if leg.precision != "fp32":
    kwargs["precision"] = leg.precision
  if leg.library:
    kwargs["library"] = leg.library
  if leg.custom_model:
    return birdnet.load_custom(
      "acoustic",
      leg.version,
      leg.backend,
      leg.custom_model,
      leg.custom_labels,
      **kwargs,
    )
  return birdnet.load("acoustic", leg.version, leg.backend, **kwargs)


def _call_kwargs(leg: Leg, on_stats: Any) -> dict[str, Any]:  # noqa: ANN401
  # Defaults stay unnamed, so a leg at the defaults runs on every version.
  kwargs: dict[str, Any] = {
    "n_producers": leg.producers,
    "batch_size": leg.batch_size,
    "device": leg.device,
  }
  if leg.workers is not None:
    kwargs["n_workers"] = leg.workers
  if leg.prefetch_ratio is not None:
    kwargs["prefetch_ratio"] = leg.prefetch_ratio
  if leg.overlap:
    kwargs["overlap_duration_s"] = leg.overlap
  if leg.speed != 1.0:
    kwargs["speed"] = leg.speed
  if leg.task == "predict":
    kwargs["top_k"] = leg.top_k
  if leg.stats:
    kwargs["show_stats"] = "progress"
    kwargs["progress_callback"] = on_stats
  return kwargs


class _Monitor:
  """Samples the machine while the timed passes run.

  Memory is summed over this process and all its children, twice: the resident
  set counts pages they share (runtime libraries, the audio ring) once per
  process, the unique set counts them not at all. What the run really occupies
  lies between the two, close to the unique set.
  """

  def __init__(self, gpu: bool, interval_s: float = 0.5) -> None:
    self._interval_s = interval_s
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._sample, daemon=True)
    self._cpu: list[float] = []
    self._rss_peak = 0
    self._uss_peak = 0
    self._uss_denied = False
    self._n_procs_peak = 0
    self._gpu_util: list[float] = []
    self._gpu_mem_peak = 0.0
    self._smi: subprocess.Popen[str] | None = None
    self._smi_thread: threading.Thread | None = None
    exe = shutil.which("nvidia-smi") if gpu else None
    if exe is not None:
      self._smi = subprocess.Popen(
        [
          exe,
          "--query-gpu=utilization.gpu,memory.used",
          "--format=csv,noheader,nounits",
          "-lms",
          "500",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
      )
      self._smi_thread = threading.Thread(target=self._read_smi, daemon=True)

  def _read_smi(self) -> None:
    assert self._smi is not None and self._smi.stdout is not None
    for line in self._smi.stdout:
      with contextlib.suppress(ValueError):
        util, mem = (float(x) for x in line.split(",")[:2])
        self._gpu_util.append(util)
        self._gpu_mem_peak = max(self._gpu_mem_peak, mem)

  def _sample(self) -> None:
    import psutil

    me = psutil.Process()
    psutil.cpu_percent(None)
    tick = 0
    while not self._stop.wait(self._interval_s):
      self._cpu.append(psutil.cpu_percent(None))
      rss, uss, n_procs = 0, 0, 0
      # The unique set is read from the page tables, so it is sampled less often.
      with_uss = tick % 4 == 0
      tick += 1
      for proc in [me, *me.children(recursive=True)]:
        with contextlib.suppress(psutil.Error):
          rss += proc.memory_info().rss
          n_procs += 1
          if with_uss:
            try:
              uss += proc.memory_full_info().uss
            except psutil.AccessDenied:
              # macOS gives another process's unique set only with its task port.
              self._uss_denied = True
      self._rss_peak = max(self._rss_peak, rss)
      self._uss_peak = max(self._uss_peak, uss)
      self._n_procs_peak = max(self._n_procs_peak, n_procs)

  def __enter__(self) -> _Monitor:
    self._thread.start()
    if self._smi_thread is not None:
      self._smi_thread.start()
    return self

  def __exit__(self, *exc: object) -> None:
    self._stop.set()
    self._thread.join(5)
    if self._smi is not None:
      self._smi.terminate()

  def summary(self) -> dict[str, Any]:
    out: dict[str, Any] = {
      "cpu_mean_pct": round(statistics.fmean(self._cpu), 1) if self._cpu else None,
      "rss_tree_peak_MiB": round(self._rss_peak / 2**20),
      "uss_tree_peak_MiB": (
        None if self._uss_denied else round(self._uss_peak / 2**20)
      ),
      "processes_peak": self._n_procs_peak,
    }
    if self._gpu_util:
      out["gpu_util_mean_pct"] = round(statistics.fmean(self._gpu_util), 1)
      out["gpu_mem_peak_MiB"] = round(self._gpu_mem_peak)
    return out


def _digest(result: Any, task: str) -> dict[str, Any]:  # noqa: ANN401
  """A few order-independent numbers of the first file's output.

  Every corpus file holds the same audio, so the first one stands for all, and
  legs that read different numbers of files stay comparable. Equal settings
  must give equal numbers whatever the pipeline does around the model; across
  backends they agree to float noise.
  """
  import numpy as np

  with contextlib.suppress(Exception):
    if task == "encode":
      emb = np.asarray(result.embeddings)[0]
      return {
        "shape": list(emb.shape),
        "abs_sum": float(np.abs(emb, dtype=np.float64).sum()),
      }
    probs = np.asarray(result.species_probs)[0]
    valid = ~np.asarray(result.species_masked)[0]
    ids = np.asarray(result.species_ids)[0]
    return {
      "shape": list(probs.shape),
      "n_detections": int(valid.sum()),
      "score_sum": float(probs[valid].astype(np.float64).sum()),
      "species_id_sum": int(ids[valid].astype(np.int64).sum()),
    }
  return {}


def _n_segments(result: Any, task: str) -> int:  # noqa: ANN401
  # Corpus files are equally long, so files x segments-per-file is exact.
  shape = (result.embeddings if task == "encode" else result.species_probs).shape
  return int(shape[0] * shape[1])


def _stats_to_dict(stats: Any) -> dict[str, Any] | None:  # noqa: ANN401
  if stats is None:
    return None
  with contextlib.suppress(Exception):
    flat = dataclasses.asdict(stats)
    return json.loads(json.dumps(flat, default=str))
  return None


def _timed(fn: Any, *args: Any, **kwargs: Any) -> tuple[float, Any]:  # noqa: ANN401
  start = time.perf_counter()
  result = fn(*args, **kwargs)
  return time.perf_counter() - start, result


def _audio_seconds(files: list[str]) -> float:
  import soundfile as sf

  # Hard links of one file: one header read is enough for equally long files.
  info = sf.info(files[0])
  return len(files) * info.frames / info.samplerate


def run_leg(args: argparse.Namespace) -> dict[str, Any]:
  import psutil

  leg = _leg_from_args(args)
  corpus = Path(args.corpus)
  all_files = _corpus_files(corpus)
  busy_before = psutil.cpu_percent(interval=1.0)

  last_stats: list[Any] = [None]

  def on_stats(stats: Any) -> None:  # noqa: ANN401
    last_stats[0] = stats

  load_s, model = _timed(_load_model, leg)
  kwargs = _call_kwargs(leg, on_stats)
  if "top_k" in kwargs:
    # A custom classifier can have fewer classes than the usual top five.
    kwargs["top_k"] = min(kwargs["top_k"], getattr(model, "n_species", leg.top_k))
  one_shot = getattr(model, leg.task)
  open_session = getattr(model, f"{leg.task}_session", None)
  log = lambda msg: print(f"  [{leg.name}] {msg}", file=sys.stderr, flush=True)  # noqa: E731
  log(f"model loaded in {load_s:.1f} s")

  def pick_files(run: Any) -> tuple[list[str], float, Any]:  # noqa: ANN401
    """Warm up, then grow the pass until it takes long enough to time.

    Every step is a whole pass, so the size is settled on passes that already
    ran warm: model load and first-shape set-up (seconds on a GPU) inflate only
    the first ones, which then merely grow the pass less.
    """
    n = min(len(all_files), max(2, leg.producers))
    first_s, result = _timed(run, all_files[:n])
    log(f"first run: {n} file(s) in {first_s:.2f} s")
    if args.files_per_pass:
      return all_files[: args.files_per_pass], first_s, result
    while True:
      seconds, result = _timed(run, all_files[:n])
      if seconds >= 0.7 * args.pass_seconds or n == len(all_files):
        break
      n = min(len(all_files), math.ceil(n * args.pass_seconds / seconds))
    if seconds < 0.7 * args.pass_seconds:
      log(
        f"corpus too small for a {args.pass_seconds:g} s pass: all "
        f"{len(all_files)} files take {seconds:.1f} s; build it with more --files"
      )
    return all_files[:n], first_s, result

  record: dict[str, Any] = {
    "schema": SCHEMA,
    "label": args.label,
    "leg": leg.name,
    "spec": dataclasses.asdict(leg),
    # Not datetime.UTC: this file also runs on the Python 3.10 of old versions.
    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # noqa: UP017
    "machine": _machine(),
    "software": _software(),
    "corpus": _corpus_id(corpus),
    "busy_before_pct": busy_before,
    "load_s": round(load_s, 3),
  }

  steady: list[float] = []
  result: Any = None
  if open_session is not None:
    record["mode"] = "session"
    enter_start = time.perf_counter()
    with open_session(**kwargs) as session:
      record["session_enter_s"] = round(time.perf_counter() - enter_start, 3)
      files, warm_s, result = pick_files(session.run)
      record["first_run_s"] = round(warm_s, 3)
      with _Monitor(gpu="GPU" in leg.device) as monitor:
        for i in range(args.repeats):
          seconds, result = _timed(session.run, files)
          steady.append(seconds)
          log(f"steady {i + 1}/{args.repeats}: {len(files)} files in {seconds:.2f} s")
      record["monitor"] = monitor.summary()
      exit_start = time.perf_counter()
    record["session_exit_s"] = round(time.perf_counter() - exit_start, 3)
    record["pipeline_stats"] = _stats_to_dict(last_stats[0])
  else:
    record["mode"] = "predict"
    files, warm_s, result = pick_files(lambda f: one_shot(f, **kwargs))
    record["first_run_s"] = round(warm_s, 3)

  # A whole call per pass: process start-up and model load included, which is
  # what a script calling predict() once, or one analysis in a GUI, pays.
  calls: list[float] = []
  with _Monitor(gpu="GPU" in leg.device) as monitor:
    for i in range(args.oneshot if steady else args.repeats):
      seconds, result = _timed(one_shot, files, **kwargs)
      calls.append(seconds)
      log(f"one-shot {i + 1}: {len(files)} files in {seconds:.2f} s")
  if not steady:
    record["monitor"] = monitor.summary()

  timed = steady or calls
  n_segments = _n_segments(result, leg.task)
  audio_s = _audio_seconds(files) / leg.speed
  fastest = min(timed)
  record.update(
    {
      "n_files": len(files),
      "n_segments": n_segments,
      "audio_s": round(audio_s, 1),
      "repeats": len(timed),
      "seconds_all": [round(t, 3) for t in timed],
      "seconds_min": round(fastest, 3),
      "seconds_median": round(statistics.median(timed), 3),
      "seconds_max": round(max(timed), 3),
      "segments_per_second": round(n_segments / fastest, 1),
      "realtime_factor": round(audio_s / fastest, 1),
      "oneshot_seconds_min": round(min(calls), 3) if calls else None,
      "startup_s": round(min(calls) - fastest, 3) if calls and steady else None,
      "digest": _digest(result, leg.task),
    }
  )
  return record


def _result_path(out: Path, label: str, host: str, leg_name: str, corpus: Path) -> Path:
  return out / f"{label}--{host}--{leg_name}@{corpus.name}.json"


def cmd_run(args: argparse.Namespace) -> None:
  record = run_leg(args)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  path = _result_path(
    out, args.label, record["machine"]["host"], record["leg"], Path(args.corpus)
  )
  path.write_text(json.dumps(record, indent=2), encoding="utf-8")
  print(
    f"{record['leg']}: {record['segments_per_second']} seg/s, "
    f"{record['realtime_factor']} xRT -> {path}"
  )


# ------------------------------------------------------------------------- suite

# Suite corpora: tag -> (format, sample rate, files factor, minutes factor).
# The factors scale --files and --minutes, so every corpus holds the same audio.
SUITE_CORPORA: dict[str, tuple[str, int, float, float]] = {
  "wav48k": ("wav", 48_000, 1, 1),
  "wav32k": ("wav", 32_000, 1, 1),
  "wav44k": ("wav", 44_100, 1, 1),
  "flac48k": ("flac", 48_000, 1, 1),
  "mp3_48k": ("mp3", 48_000, 1, 1),
  "wav48k-short": ("wav", 48_000, 40, 1 / 40),
}


def _worker_steps(physical: int, logical: int) -> list[int]:
  steps = {1, 2, 4, 8, 16, physical // 2, physical, logical}
  return sorted(w for w in steps if 1 <= w <= logical)


def suite_legs(names: list[str], physical: int, logical: int, gpu: bool) -> list[Leg]:
  """The legs of the named suites, in the order they run."""
  expanded: list[str] = []
  for name in names:
    if name == "cpu":
      expanded += [
        "cpu-backends",
        "cpu-workers",
        "cpu-batch",
        "cpu-producers",
        "cpu-inputs",
        "cpu-encode",
      ]
    elif name == "full":
      expanded += ["cpu", "gpu"]
      return suite_legs(expanded, physical, logical, gpu)
    else:
      expanded.append(name)

  onnx = Leg(version="3.0", backend="onnx")
  tflite = Leg(version="2.4", backend="tf")
  legs: list[Leg] = []
  for name in expanded:
    if name == "quick":
      legs += [onnx, tflite]
      if gpu:
        legs.append(dataclasses.replace(onnx, device="GPU", workers=1, batch_size=64))
    elif name == "cpu-backends":
      legs += [
        onnx,
        Leg(version="3.0", backend="tf"),
        Leg(version="3.0", backend="tf", library="litert"),
        Leg(version="3.0", backend="pt"),
        Leg(version="3.0", backend="pb"),
        tflite,
        Leg(version="2.4", backend="tf", library="litert"),
        Leg(version="2.4", backend="pb"),
        Leg(version="perch", backend="pb"),
      ]
    elif name == "cpu-workers":
      for base in (onnx, tflite):
        legs += [
          dataclasses.replace(base, workers=w) for w in _worker_steps(physical, logical)
        ]
    elif name == "cpu-batch":
      # The last one leaves the worker count to the library at a large batch.
      legs += [dataclasses.replace(onnx, batch_size=b) for b in (4, 16, 64)]
      legs += [dataclasses.replace(tflite, batch_size=b) for b in (4, 16)]
      # Few processes with many threads each, against many single-threaded ones.
      legs += [
        dataclasses.replace(onnx, workers=1, batch_size=16),
        dataclasses.replace(onnx, workers=1, batch_size=64),
        dataclasses.replace(onnx, workers=2, batch_size=8),
        dataclasses.replace(onnx, workers=physical, batch_size=64),
      ]
    elif name == "cpu-producers":
      legs += [dataclasses.replace(onnx, producers=p) for p in (2, 4)]
      legs += [dataclasses.replace(tflite, producers=p) for p in (2, 4)]
    elif name == "cpu-inputs":
      for corpus in ("wav32k", "wav44k", "flac48k", "mp3_48k", "wav48k-short"):
        legs.append(dataclasses.replace(onnx, corpus=corpus))
      legs.append(dataclasses.replace(tflite, corpus="wav32k"))
      legs.append(dataclasses.replace(onnx, overlap=1.5))
      legs.append(dataclasses.replace(onnx, stats=True))
    elif name == "cpu-encode":
      legs += [
        dataclasses.replace(onnx, task="encode"),
        dataclasses.replace(tflite, task="encode"),
      ]
    elif name == "gpu":
      if not gpu:
        print(
          "suite 'gpu' skipped: nvidia-smi finds no GPU (--gpu yes runs it anyway)",
          file=sys.stderr,
        )
        continue
      on_gpu = dataclasses.replace(onnx, device="GPU", workers=1)
      legs += [dataclasses.replace(on_gpu, batch_size=b) for b in (1, 16, 64, 128)]
      legs += [dataclasses.replace(on_gpu, batch_size=64, producers=p) for p in (2, 4)]
      legs += [
        dataclasses.replace(on_gpu, batch_size=128, producers=2, workers=2),
        dataclasses.replace(on_gpu, batch_size=64, producers=2, corpus="wav32k"),
        dataclasses.replace(on_gpu, batch_size=64, producers=2, precision="fp16"),
        dataclasses.replace(on_gpu, backend="pt", batch_size=64, producers=2),
        # The 3.0 SavedModel has a static batch of one segment.
        dataclasses.replace(on_gpu, backend="pb"),
      ]
      for batch_size in (64, 256):
        legs.append(
          dataclasses.replace(
            tflite, backend="pb", device="GPU", workers=1, batch_size=batch_size,
            producers=4,
          )
        )  # fmt: skip
      legs.append(
        Leg(
          version="perch", backend="pb", device="GPU", workers=1, batch_size=64,
          producers=4,
        )
      )  # fmt: skip
    else:
      raise SystemExit(f"unknown suite '{name}'")
  unique: dict[tuple[str, str], Leg] = {}
  for leg in legs:
    unique.setdefault((leg.name, leg.corpus), leg)
  return list(unique.values())


def _kill_tree(pid: int) -> None:
  import psutil

  with contextlib.suppress(psutil.Error):
    parent = psutil.Process(pid)
    procs = [*parent.children(recursive=True), parent]
    for proc in procs:
      with contextlib.suppress(psutil.Error):
        proc.kill()
    psutil.wait_procs(procs, timeout=10)


def _leg_command(
  leg: Leg, corpus: Path, label: str, out: Path, args: argparse.Namespace
) -> list[str]:
  return [
    sys.executable, str(Path(__file__).resolve()), "run",
    "--corpus", str(corpus),
    "--label", label,
    "--out", str(out),
    "--repeats", str(args.repeats),
    "--oneshot", str(args.oneshot),
    "--pass-seconds", str(args.pass_seconds),
    *leg.cli(),
  ]  # fmt: skip


def _run_leg(command: list[str], timeout: float) -> tuple[str, str, str | None]:
  """Run one leg to its end; returns its output and what went wrong, if anything.

  A leg is its own process: runtimes cannot be unloaded, a crash or a hang must
  not take the suite with it, and a hung leg is killed with its tree.
  """
  proc = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
  )
  try:
    stdout, stderr = proc.communicate(timeout=timeout)
    error = None if proc.returncode == 0 else f"exit code {proc.returncode}"
  except subprocess.TimeoutExpired:
    _kill_tree(proc.pid)
    try:
      stdout, stderr = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
      # A process that left the tree still holds the pipes.
      stdout, stderr = "", ""
    error = f"no result within {timeout:g} s (killed)"
  return stdout, stderr, error


def cmd_suite(args: argparse.Namespace) -> None:
  import psutil

  work = Path(args.work)
  out = Path(args.out) if args.out else work / "results"
  corpus_root = work / "corpus"
  out.mkdir(parents=True, exist_ok=True)
  machine = _machine()

  if args.legs:
    specs = json.loads(Path(args.legs).read_text(encoding="utf-8"))
    legs = [Leg(**spec) for spec in specs]
  else:
    legs = suite_legs(
      args.suite.split(","),
      machine["physical_cores"] or 1,
      machine["logical_cores"] or 1,
      gpu=bool(machine["gpus"]) if args.gpu == "auto" else args.gpu == "yes",
    )

  corpora: dict[str, Path | None] = {}
  for tag in dict.fromkeys(leg.corpus for leg in legs):
    fmt, sr, files_factor, minutes_factor = SUITE_CORPORA[tag]
    try:
      corpora[tag] = build_corpus(
        Path(args.seed),
        corpus_root,
        files=round(args.files * files_factor),
        minutes=args.minutes * minutes_factor,
        sr=sr,
        fmt=fmt,
      )
    except Exception as error:  # noqa: BLE001
      print(f"corpus '{tag}' unavailable ({error}); its legs are skipped")
      corpora[tag] = None

  print(f"{len(legs)} legs, label '{args.label}', results in {out}")
  pending = [
    (leg, corpus)
    for leg in legs
    if (corpus := corpora[leg.corpus]) is not None
    and (
      args.force
      or not _result_path(out, args.label, machine["host"], leg.name, corpus).is_file()
    )
  ]
  if args.warmup_legs and pending:
    leg, corpus = pending[0]
    with tempfile.TemporaryDirectory() as scratch:
      command = _leg_command(leg, corpus, "warmup", Path(scratch), args)
      for index in range(1, args.warmup_legs + 1):
        _, _, error = _run_leg(command, args.leg_timeout)
        outcome = "done, result dropped" if error is None else f"FAILED, {error}"
        print(f"warm-up {index}/{args.warmup_legs} {leg.name}: {outcome}", flush=True)
  for index, leg in enumerate(legs, 1):
    corpus = corpora[leg.corpus]
    if corpus is None:
      continue
    path = _result_path(out, args.label, machine["host"], leg.name, corpus)
    prefix = f"[{index}/{len(legs)}] {leg.name}@{corpus.name}"
    if path.is_file() and not args.force:
      print(f"{prefix}: already measured, skipped")
      continue
    command = _leg_command(leg, corpus, args.label, out, args)
    start = time.perf_counter()
    stdout, stderr, error = _run_leg(command, args.leg_timeout)
    elapsed = time.perf_counter() - start
    if error is None:
      # The leg's own line is the last one: the library prints above it.
      summary = stdout.strip().splitlines()[-1].split(" -> ")[0].split(": ", 1)[-1]
      print(f"{prefix}: {summary}  ({elapsed:.0f} s)", flush=True)
      continue
    tail = "\n".join(stderr.strip().splitlines()[-25:])
    print(f"{prefix}: FAILED, {error}\n{tail}", flush=True)
    failure = {
      "schema": SCHEMA,
      "label": args.label,
      "leg": leg.name,
      "spec": dataclasses.asdict(leg),
      "machine": machine,
      "corpus": _corpus_id(corpus),
      "error": error,
      "stderr_tail": tail,
    }
    path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
    # Children of a killed leg would skew every leg after it.
    for child in psutil.Process().children(recursive=True):
      _kill_tree(child.pid)

  print()
  cmd_compare(argparse.Namespace(results=str(out), baseline=None, md=False, csv=None))


# ----------------------------------------------------------------------- compare


def _load_records(results: Path) -> list[dict[str, Any]]:
  records = []
  for path in sorted(results.rglob("*.json")):
    with contextlib.suppress(Exception):
      record = json.loads(path.read_text(encoding="utf-8"))
      if "label" not in record:
        continue
      if "leg" not in record:  # schema 1: one 2.4 leg per file
        settings = record.get("settings", {})
        record["leg"] = (
          f"2.4-{record.get('backend', 'tf')}-cpu-w{settings.get('n_workers')}"
          f"-b{settings.get('batch_size')}-p{settings.get('n_producers')}"
        )
        record["mode"] = "predict"
      record.setdefault("corpus", "?")
      records.append(record)
  return records


def _call_seconds(record: dict[str, Any]) -> float | None:
  """Seconds of one whole call, whichever way the record was timed."""
  if record.get("mode") == "predict":
    return record["seconds_min"]
  return record.get("oneshot_seconds_min")


def _delta(record: dict[str, Any], base: dict[str, Any]) -> str:
  """Throughput against the baseline, between like and like.

  A version without sessions is timed per call, model load included; against a
  session record only the whole calls compare, per segment.
  """
  if record.get("mode") == base.get("mode"):
    ours, theirs, note = record["seconds_min"], base["seconds_min"], ""
  else:
    ours, theirs, note = _call_seconds(record), _call_seconds(base), " (calls)"
  if not ours or not theirs:
    return ""
  # From the pass times: the stored seg/s is rounded to one decimal.
  ratio = (record["n_segments"] / ours) / (base["n_segments"] / theirs)
  return f"{(ratio - 1) * 100:+.1f}%{note}"


def _memory(monitor: dict[str, Any]) -> str:
  if monitor.get("uss_tree_peak_MiB"):
    return f"{monitor['uss_tree_peak_MiB']:.0f}"
  if monitor.get("rss_tree_peak_MiB"):
    return f"{monitor['rss_tree_peak_MiB']:.0f} rss"
  return ""


def _flags(record: dict[str, Any]) -> str:
  flags = []
  low, high = record["seconds_min"], record["seconds_max"]
  if low and (high - low) / low > 0.10:
    flags.append("noisy")
  if (record.get("busy_before_pct") or 0) > 15:
    flags.append("busy-machine")
  if record.get("mode") == "predict":
    flags.append("per-call")
  return ",".join(flags)


def cmd_compare(args: argparse.Namespace) -> None:
  records = _load_records(Path(args.results))
  if not records:
    raise SystemExit(f"no result files under {args.results}")

  def key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (record["leg"], record["corpus"], record["machine"].get("host", "?"))

  baselines = {
    key(r): r for r in records if r["label"] == args.baseline and "error" not in r
  }
  header = [
    "leg", "corpus", "host", "label", "seg/s", "xRT", "pass s", "call s",
    "startup s", "cpu %", "gpu %", "mem MiB", "vs base", "flags",
  ]  # fmt: skip
  rows: list[list[str]] = []
  for record in sorted(records, key=lambda r: (*key(r), r["label"])):
    row = [
      record["leg"],
      record["corpus"].split("#")[0],
      key(record)[2],
      record["label"],
    ]
    if "error" in record:
      rows.append([*row, "FAILED", *[""] * 8, record["error"]])
      continue
    monitor = record.get("monitor") or {}
    base = baselines.get(key(record))
    delta = _delta(record, base) if base is not None and base is not record else ""

    def cell(value: object, digits: int = 1) -> str:
      return "" if value is None else f"{value:.{digits}f}"

    rows.append(
      [
        *row,
        cell(record["segments_per_second"]),
        cell(record["realtime_factor"], 0),
        cell(record["seconds_min"], 2),
        cell(record.get("oneshot_seconds_min"), 2),
        cell(record.get("startup_s"), 2),
        cell(monitor.get("cpu_mean_pct"), 0),
        cell(monitor.get("gpu_util_mean_pct"), 0),
        _memory(monitor),
        delta,
        _flags(record),
      ]
    )

  if args.csv:
    import csv

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
      csv.writer(f).writerows([header, *rows])
  widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
  if args.md:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
      print("| " + " | ".join(row) + " |")
    return
  for row in [header, *rows]:
    cells = zip(row, widths, strict=True)
    aligned = (c.ljust(w) if i < 4 else c.rjust(w) for i, (c, w) in enumerate(cells))
    print("  ".join(aligned))


# -------------------------------------------------------------------------- main


def _add_leg_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--version", default="3.0", choices=["2.4", "3.0", "perch"])
  parser.add_argument("--backend", default="onnx", choices=["tf", "pb", "onnx", "pt"])
  parser.add_argument("--library", default=None, choices=["tflite", "litert"])
  parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
  parser.add_argument("--device", default="CPU", help="CPU, GPU or GPU:<index>")
  parser.add_argument("--task", default="predict", choices=["predict", "encode"])
  parser.add_argument("--workers", type=int, default=None, help="default: library's")
  parser.add_argument("--producers", type=int, default=1)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--prefetch-ratio", type=int, default=None)
  parser.add_argument("--overlap", type=float, default=0.0)
  parser.add_argument("--speed", type=float, default=1.0)
  parser.add_argument("--top-k", type=int, default=5)
  parser.add_argument(
    "--stats",
    action="store_true",
    help="run with show_stats='progress' and keep the pipeline's own statistics",
  )
  parser.add_argument(
    "--custom-model", default=None, help="a custom classifier, read by load_custom"
  )
  parser.add_argument("--custom-labels", default=None, help="its label file")


def _add_timing_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument("--repeats", type=int, default=3, help="timed passes")
  parser.add_argument(
    "--oneshot", type=int, default=1, help="whole predict() calls timed on top"
  )
  parser.add_argument("--pass-seconds", type=float, default=PASS_SECONDS)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = parser.add_subparsers(dest="cmd", required=True)

  e = sub.add_parser("env", help="print the machine and the installed versions")
  e.set_defaults(func=cmd_env)

  c = sub.add_parser("corpus", help="build a reproducible corpus from a seed file")
  c.add_argument("--seed", required=True, help="an audio file to tile")
  c.add_argument("--out", default="./corpus")
  c.add_argument("--files", type=int, default=240)
  c.add_argument("--minutes", type=float, default=10)
  c.add_argument("--sr", type=int, default=48_000)
  c.add_argument("--format", default="wav", choices=["wav", "flac", "mp3", "ogg"])
  c.set_defaults(func=cmd_corpus)

  r = sub.add_parser("run", help="measure one leg in this process")
  r.add_argument("--corpus", required=True)
  r.add_argument("--label", required=True, help="e.g. v1.5.1 or main-f51ce46")
  r.add_argument("--out", default="./results")
  r.add_argument(
    "--files-per-pass", type=int, default=None, help="default: sized by rate"
  )
  _add_leg_arguments(r)
  _add_timing_arguments(r)
  r.set_defaults(func=cmd_run)

  s = sub.add_parser("suite", help="measure a set of legs, one process per leg")
  s.add_argument("--suite", default="quick", help="comma list of: quick, cpu, gpu, "
                 "full, cpu-backends, cpu-workers, cpu-batch, cpu-producers, "
                 "cpu-inputs, cpu-encode")  # fmt: skip
  s.add_argument("--legs", default=None, help="JSON file with a list of leg specs")
  s.add_argument("--label", required=True)
  s.add_argument("--seed", required=True, help="an audio file to tile")
  s.add_argument("--work", default="./bench-work", help="holds corpus/ and results/")
  s.add_argument("--out", default=None, help="default: <work>/results")
  s.add_argument("--files", type=int, default=240)
  s.add_argument("--minutes", type=float, default=10)
  s.add_argument("--leg-timeout", type=float, default=1800)
  s.add_argument(
    "--warmup-legs",
    type=int,
    default=0,
    help="run the first leg this often before the suite and drop the results",
  )
  s.add_argument(
    "--gpu",
    default="auto",
    choices=["auto", "yes", "no"],
    help="run the GPU legs; auto: where nvidia-smi lists a GPU",
  )
  s.add_argument("--force", action="store_true", help="measure legs again")
  _add_timing_arguments(s)
  s.set_defaults(func=cmd_suite)

  k = sub.add_parser("compare", help="tabulate the collected results")
  k.add_argument("--results", default="./results")
  k.add_argument("--baseline", default=None, help="label the delta column refers to")
  k.add_argument("--md", action="store_true", help="print a markdown table")
  k.add_argument("--csv", default=None, help="also write the table to this file")
  k.set_defaults(func=cmd_compare)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
