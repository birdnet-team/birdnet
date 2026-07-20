# ruff: noqa: T201, ANN001
"""Throughput benchmark for the per-file completion callback (``on_file_complete``).

Compares prediction throughput across three conditions on the same synthetic
workload to show that enabling the callback does not regress throughput:

  * baseline  – ``on_file_complete=None`` (feature fully inert)
  * noop      – a callback that does nothing (isolates pipeline overhead:
                markers, per-file bookkeeping, slice copies, result construction)
  * persist   – a callback that writes each file's result to CSV (the resumable
                analysis use case; includes real disk I/O)

Run:
    python benchmarks/on_file_complete_benchmark.py \
        --files 48 --seconds 15 --reps 3 --workers 8
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from birdnet.model_loader import load

SAMPLE_RATE = 48_000


def make_audio_files(out_dir: Path, n_files: int, seconds: float) -> list[str]:
  rng = np.random.default_rng(1234)
  n_samples = int(seconds * SAMPLE_RATE)
  paths: list[str] = []
  for i in range(n_files):
    # Low-amplitude noise -> a few detections per file, realistic write sizes.
    audio = (rng.standard_normal(n_samples) * 0.05).astype(np.float32)
    p = out_dir / f"bench_{i:04d}.wav"
    sf.write(p, audio, SAMPLE_RATE)
    paths.append(str(p))
  return paths


def run_once(model, files, *, on_file_complete, workers, batch_size, top_k) -> float:
  start = time.perf_counter()
  with model.predict_session(
    n_workers=workers,
    batch_size=batch_size,
    top_k=top_k,
    on_file_complete=on_file_complete,
  ) as session:
    session.run(files)
  return time.perf_counter() - start


def summarise(name: str, times: list[float], total_audio_s: float) -> dict:
  best = min(times)
  median = statistics.median(times)
  return {
    "name": name,
    "median_s": median,
    "best_s": best,
    "xrt_median": total_audio_s / median,
    "xrt_best": total_audio_s / best,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--files", type=int, default=48)
  parser.add_argument("--seconds", type=float, default=15.0)
  parser.add_argument("--reps", type=int, default=3)
  parser.add_argument("--workers", type=int, default=8)
  parser.add_argument("--batch-size", type=int, default=8)
  parser.add_argument("--top-k", type=int, default=5)
  args = parser.parse_args()

  total_audio_s = args.files * args.seconds

  with tempfile.TemporaryDirectory() as tmp:
    tmp_dir = Path(tmp)
    csv_dir = tmp_dir / "csv"
    csv_dir.mkdir()
    print(
      f"Generating {args.files} files x {args.seconds}s "
      f"({total_audio_s / 60:.1f} min audio)..."
    )
    files = make_audio_files(tmp_dir, args.files, args.seconds)

    model = load("acoustic", "2.4", "tf", precision="fp32", library="tflite")

    def persist(result) -> None:  # noqa: ANN001
      name = Path(str(result.inputs[0])).stem
      result.to_csv(csv_dir / f"{name}.csv", silent=True)

    conditions = {
      "baseline (no callback)": None,
      "noop callback": lambda _r: None,
      "persist callback (to_csv)": persist,
    }

    common = {
      "workers": args.workers,
      "batch_size": args.batch_size,
      "top_k": args.top_k,
    }

    # Warm up (model load, tflite graph, process spawn) — not measured.
    print("Warming up...")
    run_once(model, files, on_file_complete=None, **common)

    # Interleave conditions across reps so system drift (thermal, scheduling)
    # affects every condition equally rather than biasing one block.
    times: dict[str, list[float]] = {name: [] for name in conditions}
    for r in range(args.reps):
      for name, cb in conditions.items():
        dt = run_once(model, files, on_file_complete=cb, **common)
        times[name].append(dt)
        print(f"  rep {r + 1}/{args.reps}  {name:28s}: {dt:6.2f}s")

    results = [
      summarise(name, times[name], total_audio_s) for name in conditions
    ]

    baseline = results[0]
    print("\n" + "=" * 78)
    print(
      f"Workload: {args.files} files x {args.seconds}s = {total_audio_s / 60:.1f} min "
      f"| workers={args.workers} batch={args.batch_size} top_k={args.top_k} "
      f"| reps={args.reps}"
    )
    print("=" * 78)
    header = (
      f"{'condition':30s}{'median s':>11s}{'best s':>10s}"
      f"{'xRT (best)':>13s}{'vs baseline':>14s}"
    )
    print(header)
    print("-" * 78)
    for res in results:
      delta = (res["best_s"] - baseline["best_s"]) / baseline["best_s"] * 100
      print(
        f"{res['name']:30s}{res['median_s']:>11.2f}{res['best_s']:>10.2f}"
        f"{res['xrt_best']:>12.1f}x{delta:>+13.1f}%"
      )
    print("=" * 78)
    print(
      "Lower 'vs baseline' magnitude = less overhead. Positive = slower than "
      "baseline; within run-to-run noise (~a few %) means no regression."
    )


if __name__ == "__main__":
  main()
