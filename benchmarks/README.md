# Cross-version benchmark harness

`bench_birdnet.py` answers one question: did inference throughput change between
two birdnet versions? It is the plausibility check to run before a release
(nothing broke that made it much slower), and it works across machines.

It is deliberately not the same tool as the shipped `birdnet-benchmark` CLI:
that CLI profiles the *installed* birdnet in depth (buffer, producer and worker
timings) and follows the current API. This script touches only `birdnet.load()`
and `model.predict()`, which are stable from 0.2.16 onward, so the identical
file runs inside any version's virtualenv — including versions that predate the
CLI.

## Workflow

Once per machine, build the corpus (deterministic from the seed, so it can be
rebuilt any time):

```sh
python benchmarks/bench_birdnet.py corpus --seed example/soundscape.wav \
    --out benchmarks/corpus --files 12 --minutes 10
```

Once per version, inside that version's own virtualenv:

```sh
# a released version: pip install birdnet==<version> into a fresh venv first
python benchmarks/bench_birdnet.py run --corpus benchmarks/corpus/12x10min \
    --label v1.1.0 --workers 8 --out benchmarks/results

# the working tree: use the dev venv, label with the commit
python benchmarks/bench_birdnet.py run --corpus benchmarks/corpus/12x10min \
    --label main-4c95133 --workers 8 --out benchmarks/results
```

Then tabulate:

```sh
python benchmarks/bench_birdnet.py compare --results benchmarks/results \
    --baseline v1.1.0
```

`corpus/` and `results/` under this directory are gitignored; results are
per-machine measurements, not repo artifacts.

## Rules that keep the numbers honest

- **Same corpus, same settings, same machine** for any legs you compare
  directly. The delta column in `compare` is only meaningful within one host;
  across hosts compare xRT (real-time factor). Result filenames carry the
  hostname so several machines can be collected into one directory.
- **One leg at a time, idle machine.** Wall time is taken as the minimum over
  repeats, which forgives background noise but cannot forgive a parallel
  benchmark. A spread above 10 % is flagged `(noisy)` — rerun rather than
  interpret.
- **`--label` is the source of truth** for what was measured. The `birdnet`
  field in the JSON comes from installed distribution metadata, and an editable
  install of a work branch reports its base version. Label branch runs
  `main-<shortsha>`.
- **Old versions get their own model cache.** Model files are cached under a
  path with no release in it and (for tf/pt/onnx) identified by size alone, so
  point `BIRDNET_APP_DATA` at a separate directory when running an old version,
  or the legs can silently swap or re-download each other's models.
- The warm-up pass (discarded) pays for model download and backend init; the
  timed repeats measure steady state.

## Pre-release plausibility check

1. Build the corpus if the machine does not have one.
2. Run the previous release from PyPI in a fresh venv (label `v<version>`).
3. Run the release candidate from the working tree (label `v<next>-rc` or
   `main-<shortsha>`).
4. `compare --baseline v<version>`: the candidate should be within a few
   percent. A double-digit regression is a stop-and-investigate, not a note in
   the release.

macOS throughput baselines from the 2026-08 investigation (tflite 2.4 fp32, M1,
median xRT 173/280/294 at 2/4/8 workers) predate this schema and live outside
the repo; regenerate them with this script when next on that machine.
