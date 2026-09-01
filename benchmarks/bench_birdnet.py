#!/usr/bin/env python3
"""Time one birdnet version on a fixed corpus, and print the result as JSON.

Self-contained on purpose: it is copied onto whichever machine is being measured
and run once per version inside that version's own virtualenv. It touches only
`birdnet.load(..)` and `model.predict(..)`, which are unchanged from 0.2.16
through 1.1.0, so the same file drives every leg of the comparison. It depends
only on the stdlib plus numpy, soundfile and psutil, which every birdnet version
in scope already installs.

  # once per machine
  python bench_birdnet.py corpus --seed example/soundscape.wav --out ./corpus \
      --files 12 --minutes 10

  # once per version, in that version's venv
  python bench_birdnet.py run --corpus ./corpus/12x10min --label v1.1.0 \
      --workers 8 --repeats 4 --out results/

Wall time is reported as the *minimum* over repeats, not the mean: a loaded
machine can only ever make a run slower, so the floor is the honest estimate of
what the code costs. The spread is reported too, so a noisy machine is visible
rather than averaged away.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

SEGMENT_S = 3.0  # v2.4 segment length; used to derive segments/s


def build_corpus(args: argparse.Namespace) -> None:
  import numpy as np
  import soundfile as sf

  seed = Path(args.seed)
  data, sr = sf.read(seed, dtype="float32")
  if data.ndim > 1:
    data = data.mean(axis=1)

  seed_s = len(data) / sr
  reps = max(1, round(args.minutes * 60 / seed_s))
  tiled = np.tile(data, reps)

  target = Path(args.out) / f"{args.files}x{args.minutes}min"
  target.mkdir(parents=True, exist_ok=True)
  for i in range(args.files):
    # Distinct paths matter: birdnet de-duplicates inputs by absolute path, so
    # the same file passed N times collapses to one and the run ends early.
    sf.write(target / f"part_{i:03d}.wav", tiled, sr)

  total_min = args.files * len(tiled) / sr / 60
  print(  # noqa: T201
    f"corpus: {target}  files={args.files}  audio={total_min:.1f} min  sr={sr}",
    file=sys.stderr,
  )
  print(target)  # noqa: T201


def _birdnet_version() -> str:
  # The package exposes no __version__; the installed distribution knows. An
  # editable install of a work branch reports its base version, so the label
  # stays the source of truth for *what* was measured -- this field only
  # cross-checks that the right venv was active.
  import importlib.metadata

  try:
    return importlib.metadata.version("birdnet")
  except importlib.metadata.PackageNotFoundError:
    return "unknown"


def _dep_versions() -> dict:
  # The inference runtime dominates the measurement, so a version drift there
  # between venvs must be visible in the record.
  import importlib.metadata

  versions = {}
  for dist in ("tensorflow", "ai-edge-litert", "torch", "onnxruntime"):
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
      versions[dist] = importlib.metadata.version(dist)
  return versions


def _machine() -> dict:
  import psutil

  return {
    "host": platform.node(),
    "os": f"{platform.system()} {platform.release()}",
    "cpu": platform.processor() or platform.machine(),
    "physical_cores": psutil.cpu_count(logical=False),
    "logical_cores": psutil.cpu_count(logical=True),
    "ram_GiB": round(psutil.virtual_memory().total / 2**30, 1),
  }


def run(args: argparse.Namespace) -> None:
  import birdnet

  files = sorted(str(p) for p in Path(args.corpus).glob("*.wav"))
  if not files:
    raise SystemExit(f"no .wav files under {args.corpus}")

  model = birdnet.load(
    "acoustic", "2.4", args.backend, precision="fp32", library=args.library
  )

  kwargs = {
    "n_workers": args.workers,
    "n_producers": args.producers,
    "batch_size": args.batch_size,
    "top_k": args.top_k,
    "device": args.device,
  }

  # First pass is thrown away: it pays for model download, backend init and page
  # faults on the corpus, none of which is what we are comparing.
  durations: list[float] = []
  n_segments = 0
  for i in range(args.repeats + 1):
    start = time.perf_counter()
    result = model.predict(files, **kwargs)
    elapsed = time.perf_counter() - start
    if i == 0:
      n_segments = int(result.species_probs.shape[0] * result.species_probs.shape[1])
      print(  # noqa: T201
        f"warm-up: {elapsed:.2f} s ({n_segments} segments)", file=sys.stderr
      )
      continue
    durations.append(elapsed)
    print(f"  run {i}: {elapsed:.2f} s", file=sys.stderr)  # noqa: T201

  fastest = min(durations)
  machine = _machine()
  record = {
    "label": args.label,
    "birdnet": _birdnet_version(),
    "deps": _dep_versions(),
    "python": platform.python_version(),
    "platform": f"{platform.system()}-{platform.machine()}",
    "machine": machine,
    "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    "corpus": str(args.corpus),
    "n_files": len(files),
    "n_segments": n_segments,
    "settings": dict(kwargs),
    "backend": args.backend,
    "library": args.library,
    "repeats": args.repeats,
    "seconds_min": round(fastest, 3),
    "seconds_median": round(statistics.median(durations), 3),
    "seconds_max": round(max(durations), 3),
    "segments_per_second": round(n_segments / fastest, 1) if fastest else 0.0,
    "realtime_factor": round(n_segments * SEGMENT_S / fastest, 1) if fastest else 0.0,
  }

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)
  # Host in the filename keeps results from several machines collision-free
  # when they are collected into one directory for comparison.
  path = out_dir / f"{args.label}--{machine['host']}.json"
  path.write_text(json.dumps(record, indent=2), encoding="utf-8")
  print(json.dumps(record, indent=2))  # noqa: T201


def compare(args: argparse.Namespace) -> None:
  records = [
    json.loads(p.read_text(encoding="utf-8"))
    for p in sorted(Path(args.results).glob("*.json"))
  ]
  if not records:
    raise SystemExit(f"no result files under {args.results}")

  baseline = next((r for r in records if r["label"] == args.baseline), records[0])
  base_s = baseline["seconds_min"]

  hosts = {r.get("machine", {}).get("host", "?") for r in records}
  if len(hosts) > 1:
    print(  # noqa: T201
      "results span multiple hosts; wall seconds and the delta column are only "
      "meaningful within one host -- compare xRT across hosts",
      file=sys.stderr,
    )

  width = max(len(r["label"]) for r in records)
  hwidth = max(len(h) for h in hosts)
  print(  # noqa: T201
    f"{'label'.ljust(width)}  {'host'.ljust(hwidth)}  "
    f"{'min s':>8} {'seg/s':>8} {'xRT':>7}  vs {baseline['label']}"
  )
  for r in sorted(records, key=lambda r: r["seconds_min"]):
    host = r.get("machine", {}).get("host", "?")
    delta = (r["seconds_min"] - base_s) / base_s * 100 if base_s else 0.0
    spread = (r["seconds_max"] - r["seconds_min"]) / r["seconds_min"] * 100
    flag = "  (noisy)" if spread > 10 else ""
    print(  # noqa: T201
      f"{r['label'].ljust(width)}  {host.ljust(hwidth)}  "
      f"{r['seconds_min']:>8.2f} "
      f"{r['segments_per_second']:>8.1f} {r['realtime_factor']:>7.1f}  "
      f"{delta:+6.1f}%{flag}"
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = parser.add_subparsers(dest="cmd", required=True)

  c = sub.add_parser("corpus", help="build a reproducible corpus from a seed file")
  c.add_argument("--seed", required=True, help="a wav file to tile")
  c.add_argument("--out", default="./corpus")
  c.add_argument("--files", type=int, default=12)
  c.add_argument("--minutes", type=int, default=10)
  c.set_defaults(func=build_corpus)

  r = sub.add_parser("run", help="time one version on a corpus")
  r.add_argument("--corpus", required=True)
  r.add_argument("--label", required=True, help="e.g. v0.2.16, v1.1.0, main-4c95133")
  r.add_argument("--out", default="./results")
  r.add_argument("--workers", type=int, default=None)
  r.add_argument("--producers", type=int, default=1)
  r.add_argument("--batch-size", type=int, default=1)
  r.add_argument("--top-k", type=int, default=5)
  r.add_argument("--device", default="CPU")
  r.add_argument("--backend", default="tf")
  r.add_argument("--library", default="tflite")
  r.add_argument("--repeats", type=int, default=3)
  r.set_defaults(func=run)

  k = sub.add_parser("compare", help="tabulate the collected results")
  k.add_argument("--results", default="./results")
  k.add_argument("--baseline", default="v1.1.0")
  k.set_defaults(func=compare)

  args = parser.parse_args()
  args.func(args)


if __name__ == "__main__":
  main()
