# Benchmark harness

`bench_birdnet.py` measures the inference throughput of one birdnet install: a
single configuration (a *leg*) or a suite of them, on a corpus that every
machine rebuilds identically. It answers two questions: did a change make
inference faster or slower, and what does this machine reach with which
settings.

It is one self-contained file. It needs only the stdlib plus numpy, scipy,
soundfile and psutil, which every birdnet version installs, and it touches only
the public API (`birdnet.load`, `load_perch_v2`, `predict`/`encode` and, where
the version has them, the session methods). The same file therefore runs in the
virtualenv of any version from 0.2.16 on, and can be copied to a machine that
has no checkout.

The shipped `birdnet-benchmark` CLI is a different tool: it runs the installed
version once and shows the pipeline's own live statistics (buffer, producer and
worker timings). Use the CLI to look into one run, and this harness to compare
versions, settings and machines.

`NOTES.md` next to this file holds what the measurements found so far, the
reference numbers per machine and a checklist for the next platform.

## Quick start

```sh
# what would be measured: machine, versions, where birdnet is imported from
python benchmarks/bench_birdnet.py env

# the pre-release check: three legs, about five minutes
python benchmarks/bench_birdnet.py suite --suite quick --label main-<shortsha> \
    --seed example/soundscape.wav --work ./bench-work

# everything this machine can run: about 100 minutes
python benchmarks/bench_birdnet.py suite --suite full --label main-<shortsha> \
    --seed example/soundscape.wav --work ./bench-work

# one leg by hand
python benchmarks/bench_birdnet.py run --corpus ./bench-work/corpus/wav48k_240x10min \
    --label main-<shortsha> --version 3.0 --backend onnx --workers 8

# the table, with a delta column against a baseline label
python benchmarks/bench_birdnet.py compare --results ./bench-work/results \
    --baseline v1.5.1
```

`--work` holds `corpus/` and `results/`. Put it on a disk with room: the corpus
is one 58 MB file plus hard links to it, but on a file system without hard links
it becomes 240 copies (14 GB; lower `--files` there). A suite skips the legs it
already has a result for, so an interrupted suite is continued by running the
same command again; `--force` measures again.

## What a leg measures

Every leg runs in its own process. It loads the model, opens one session and
then:

1. **Sizes the pass.** It runs a few files, then grows the file list until one
   pass takes about `--pass-seconds` (12 s). A slow configuration reads a few
   files, a GPU reads many, and both get a pass long enough to time.
2. **Times `--repeats` passes in the open session** (`session.run`). The fastest
   is the result: *seg/s* (segments per second) and *xRT* (seconds of audio per
   second). The workers are up and the models loaded, but a pass is a whole
   run: it includes what every run costs once, such as starting and draining
   the pipeline and the last short batches. Where that cost is small the
   number is what a long run converges to; where it is not, the number depends
   on the pass length, so compare at equal `--pass-seconds` and check a
   surprising difference with a longer pass (see `NOTES.md` for an example).
3. **Times one whole call** (`model.predict` on the same files): process
   start-up and model load included. *call s* minus *pass s* is the *startup s*
   column, what every call costs on top, which is most of what a single short
   file takes.

It also records, per leg:

- **cpu %**, **gpu %**: mean utilisation during the timed passes. A GPU far
  below 100 % is waiting for the CPU side (producers, post-processing).
- **mem MiB**: peak unique memory of the process tree: what the run occupies
  on top of the pages its processes share, such as a model file they all map.
  Where the platform does not report it, the column shows the resident sum
  with `rss` behind it, which counts shared pages once per process and is
  several times larger for TensorFlow. The JSON has both.
- **digest**: detection count and score sum of the first file. A changed digest
  at equal settings means the change computes something else. An equal one is
  a plausibility check, not a proof: it is a sum over one file, and the corpus
  repeats the 40 segments of the seed.
- `load_s`, `session_enter_s`, `first_run_s`, `session_exit_s`, every pass time,
  versions of the runtimes, the commit and dirty files of an editable install.

A version without session methods is timed per call only and flagged
`per-call`. Against a record timed in a session, `compare` then sets whole
calls against whole calls and marks the delta `(calls)`.

Every file of a corpus is a hard link to one file, so after the first read it
comes from the page cache: the numbers say nothing about a slow disk.

## Suites

| suite | legs | what it answers |
|-------|------|-----------------|
| `quick` | 3.0 onnx and 2.4 tf at the defaults, 3.0 onnx on a GPU if there is one | did anything get slower |
| `cpu-backends` | every model and backend at the defaults | which backend to use |
| `cpu-workers` | worker counts from 1 to all logical cores | scaling, hyper-threads, efficiency cores |
| `cpu-batch` | batch sizes, few workers with many threads, the worker count left to the library at a large batch | what a batch costs on the CPU, and what one process with many threads reaches |
| `cpu-producers` | 2 and 4 producers | whether decoding limits |
| `cpu-inputs` | 32 and 44.1 kHz, FLAC, MP3, short files, overlap, statistics on | what the input costs |
| `cpu-encode` | embeddings | the encoding path |
| `gpu` | batch sizes, producers, one and two workers, fp16, pt, pb, Perch | GPU settings |
| `cpu`, `full` | all `cpu-*`; `cpu` plus `gpu` | |

`--legs legs.json` runs a list of your own: each entry is an object with the
fields of the `Leg` class at the top of the legs section in the script.

A custom classifier is a leg of its own, since it needs its files:

```sh
python benchmarks/bench_birdnet.py run --corpus ./bench-work/corpus/wav48k_240x10min \
    --label main-<shortsha> --version 3.0 --backend tf \
    --custom-model classifier.tflite --custom-labels classifier_Labels.txt
```

The GPU legs run where `nvidia-smi` lists a GPU; `--gpu yes` runs them anyway,
for a GPU it does not know (DirectML).

## Rules that keep the numbers honest

- **One suite at a time on an idle machine.** The result is the fastest of the
  passes, which forgives a short disturbance but not a second benchmark, a test
  suite or a build. The table flags `noisy` (passes more than 10 % apart) and
  `busy-machine` (CPU already in use before the leg): measure those again.
- **Never edit what is being measured.** A suite takes an hour, and every leg
  imports birdnet afresh. Measure an installed version, or a copy of the source
  tree selected with `PYTHONPATH` (a path there wins over an editable install),
  and copy the script itself if you keep working on it.
- **`--label` is what was measured.** The distribution version of an editable
  install is whatever it was when the install ran; the record also holds the
  commit and the dirty files, which is what to trust.
- **Compare like with like.** The delta column compares legs of the same name,
  corpus and host. Across hosts compare xRT, never seconds.
- **Same runtimes on both sides.** onnxruntime, TensorFlow and LiteRT releases
  differ by tens of percent. To compare two birdnet versions, install both with
  the same pins (`pip freeze` of one into the other).
- **Old versions get their own model cache.** Before 1.1 a model file is
  identified by its size alone, so point `BIRDNET_APP_DATA` at a separate
  directory for them, or the versions swap or re-download each other's files.
- **One venv per GPU runtime.** TensorFlow, PyTorch and onnxruntime-gpu each
  want their own CUDA and cuDNN builds; installed together, the pip packages of
  one break the GPU of another (`DNN library initialization failed`, or a
  fall back to the CPU, which birdnet reports as an error).

## Pre-release check

1. `suite --suite quick` with the previous release from PyPI in a fresh venv,
   label `v<version>`.
2. The same with the release candidate, label `main-<shortsha>`, with the same
   runtime pins.
3. `compare --baseline v<version>`: within a few percent is fine. A double-digit
   regression is a stop-and-investigate, not a note in the release. A changed
   digest is a changed result and needs an explanation before a speed-up counts.
