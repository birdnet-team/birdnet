# Benchmark notes

What the measurements found so far, the reference numbers per machine, and what
to check on the next platform. `README.md` explains the tool; this file is the
log book. Add a section per machine, do not overwrite another machine's.

## Recipe that worked (Linux, 2026-11-23)

Everything heavy lives on one roomy disk (`$W` below); the home partition only
holds the checkout.

```sh
export W=/data/bench                      # corpus, results, venvs, model cache
export BIRDNET_APP_DATA=$W/app-data       # ~6 GB once every model is loaded
export TMPDIR=$W/tmp                      # session logs, one or two files per session
export UV_CACHE_DIR=$W/uv-cache

# the tree under test, CPU legs
uv venv --python 3.13 $W/venvs/main
uv pip install --python $W/venvs/main/bin/python -e '.[tf,pt]'

# a release to compare with: same runtime pins, birdnet from PyPI
uv pip freeze --python $W/venvs/main/bin/python | grep -v -i '^-e\|^birdnet' > $W/pins.txt
uv venv --python 3.13 $W/venvs/v1.5.1
uv pip install --python $W/venvs/v1.5.1/bin/python -r $W/pins.txt birdnet==1.5.1

# measure a frozen copy of the tree, so the checkout stays editable meanwhile
mkdir -p $W/snap/work && cp -a src/birdnet $W/snap/work/ && cp benchmarks/bench_birdnet.py $W/bench.py
PYTHONPATH=$W/snap/work $W/venvs/main/bin/python $W/bench.py suite --suite cpu \
    --label work-<shortsha> --seed example/soundscape.wav --work $W
$W/venvs/v1.5.1/bin/python $W/bench.py suite --suite quick --label v1.5.1 \
    --seed example/soundscape.wav --work $W
$W/venvs/main/bin/python $W/bench.py compare --results $W/results --baseline v1.5.1
```

`env` prints where `birdnet` is imported from; check it before a long suite. The
full `cpu` suite takes about 75 minutes here, `gpu` about 20.

On Windows the same steps run in PowerShell (`$env:BIRDNET_APP_DATA = ..`,
`venvs\main\Scripts\python.exe`, `Copy-Item -Recurse` for the snapshot); the
script itself has no shell in it. NTFS allows 1023 hard links per file, so the
corpus of short files holds ten physical copies there instead of one. On macOS
there are no GPU legs (the suite skips them without `nvidia-smi`), and Perch
does not run on Intel Macs.

GPU legs need one venv per runtime, built to the CUDA the driver supports:

| runtime | venv that worked here (driver 575, CUDA 12.9) |
|---------|-----------------------------------------------|
| onnxruntime-gpu | `onnxruntime-gpu==1.26.0`, the last release built for CUDA 12; 1.27 and later need CUDA 13 and otherwise fall back to the CPU, which birdnet reports as an error. `onnxruntime` (CPU) must be uninstalled first, both provide the same module |
| PyTorch | `torch` from `https://download.pytorch.org/whl/cu128`; the default PyPI wheel (2.15) is built for CUDA 13 and reports "driver too old" |
| TensorFlow | plain `tensorflow`, which finds the system CUDA 12.9 and cuDNN 9. In a venv that also holds PyTorch, its `nvidia-*` wheels shadow those libraries and TensorFlow fails with `DNN library initialization failed` or lists no GPU at all |

The library's own `gpu` test lane skips every case unless TensorFlow sees the
GPU, so it needs TensorFlow and onnxruntime-gpu in one venv without PyTorch; its
`pt` cases then skip.

## Checklist for the next platform

1. `bench_birdnet.py env`: the right interpreter, the commit, no stray
   `OMP_NUM_THREADS` and the like in the environment.
2. `suite --suite quick` for the release the application pins and for the tree,
   `compare --baseline v<release>`: no leg more than a few percent slower.
3. `suite --suite cpu` (and `gpu`) for the tree. Compare *xRT* with the tables
   below; seconds are not comparable between machines.
4. Look at these in particular, they are where platforms differ:
   - **XNNPACK**: `3.0-tf-*` and `2.4-tf-*` legs against the release before the
     delegate, with the same fixed worker count on both sides. The gain is the
     platform's: a factor of 2.1 to 3.1 for 3.0 and 7 to 32 % for 2.4 on Linux
     x86-64, 3.0 to 3.3 and 69 to 91 % on Windows x86-64, about 5 % and nothing
     on an Apple M1. The memory is the same on all three: 50 to 60 MiB more per
     worker for 2.4 and about 300 for 3.0, which is why #185 leaves the delegate
     off on Apple silicon. The digest of a `tf` leg sits 3e-4 to 4e-4 below the
     `onnx` leg's, with and without the delegate.
   - **Worker scaling** (`cpu-workers`): where throughput stops growing. Here
     the 3.0 ONNX model stops at 4 to 6 workers of 12 cores, because memory
     bandwidth is used up; a machine with more memory channels per core (Apple
     silicon, servers) should scale further, and hybrid CPUs get little from
     their efficiency cores.
   - **One `onnx` worker with a large batch** (`3.0-onnx-cpu-w1-b64-p1`)
     against a worker per core (`3.0-onnx-cpu-w<cores>-b1-p1`), on the branch
     of #182, where that worker runs a row of the batch on every thread: 3 %
     faster on magpie, level on the M1, 3 to 4 % slower on the Windows laptop,
     in a sixth to an eighth of the memory. That is a way to save memory, not a
     default. The first version of #182 chose it by itself from four rows per
     core on, which is why the `wauto-b64` legs in the sections below ran one
     worker; on the branch as it is now they run a worker per core. One pair of
     legs does not resolve 3 %: run both several times in alternating order.
   - **startup s**: process start is cheap on Linux. On Windows (spawn, virus
     scanner) and macOS it decides how long a single short file takes.
   - **uss MiB** per worker against the estimates in
     `src/birdnet/acoustic/inference/memory.py`; they were measured on Linux.
     The two laptops found two that were off, which #184 corrects. On Windows
     a TFLite worker in TensorFlow's interpreter holds the model file once
     more: 900 MiB for 3.0 where 629 were planned. And every row of a batch was
     planned at 12 to 15 MiB, which a batch run row by row does not hold: the
     largest of 12 workers on magpie grows by 11 MiB from batch size 1 to 64
     (2.4 tflite and litert) and not at all for 3.0 onnx. The 4.5 MiB per row
     in the M1's footprints match the audio ring, shared memory that a
     footprint or a resident set counts in every worker that touches it (see
     the M1's memory section). macOS does not hand out the unique set of a
     child process: the column shows the resident sum there.
   - **A laptop runs its first legs fast.** Both laptops ran the first leg
     after a pause 10 to 15 % above the same leg later in a suite, and the
     fastest of three passes keeps that. Start a suite with `--warmup-legs 2`
     and compare two settings in alternating order.
   - **Producers**: one producer decodes FLAC and MP3 at 300 to 400 segments
     per second here and WAV at 1000 to 4400. A leg whose *cpu %* or *gpu %* is
     low while workers wait is producer-bound: compare its `p2` and `p4` legs.
   - **GPU, Windows**: DirectML takes the place of CUDA. The short-batch
     padding of the `onnx` backend applies to it as well, but its effect there
     is unmeasured.
5. Run the library's test lanes for the code the measurements touched
   (`AGENTS.md`); a faster number from a broken result is not a result.

## magpie: Linux, i7-12700F (8 P + 4 E cores, 20 threads), 31 GiB, RTX 3080 Ti

Measured 2026-11-23, Python 3.13, onnxruntime 1.32.0 (GPU legs: onnxruntime-gpu
1.26.0), TensorFlow 2.21.0, LiteRT 2.4.0, PyTorch 2.15.0 (GPU: 2.11.0+cu128),
numpy 2.5.5, scipy 1.18.1, libsndfile 1.2.2. Corpus `wav48k_240x10min` (the
example soundscape tiled to 10-minute files, 48 kHz 16-bit mono) unless a leg
names another. `main` is f51ce46 (1.5.1 plus six fixes); `tree` is main with the
changes listed under "What changed" below. seg/s is the fastest of three passes
of about 12 s in a warm session; xRT = seconds of audio per second. Passes of
48 s gave the same numbers on the tree (3.0 onnx 71.2, 2.4 tf 233.4 on the CPU,
3.0 onnx 654.6 on the GPU), so they stand for long runs too.

### CPU, defaults (one worker per physical core, one thread each, batch size 1)

| model, backend | main seg/s | tree seg/s | tree xRT | change | startup s | memory GiB |
|---|---|---|---|---|---|---|
| 3.0 onnx | 71.9 | 70.9 | 213 | same | 1.6 | 5.8 |
| 3.0 tf (TensorFlow Lite) | 25.8 | 54.3 | 163 | +110 % | 3.0 | 6.9 |
| 3.0 tf, `library="litert"` | 20.2 | 62.4 | 187 | +209 % | 1.2 | 4.5 |
| 3.0 pt | 33.9 | 33.8 | 102 | same | 4.5 | 8.6 |
| 3.0 pb | 7.3 | 7.3 | 22 | unchanged code | 20.4 |  |
| 2.4 tf (TensorFlow Lite) | 218.9 | 235.0 | 705 | +7 % | 3.0 | 3.6 |
| 2.4 tf, `library="litert"` | 186.2 | 245.2 | 736 | +32 % | 0.6 | 1.2 |
| 2.4 pb | 29.5 | 29.5 | 88 | same | 8.4 | 8.0 |
| Perch v2 pb (5 s segments) | 6.6 | 6.6 | 33 | unchanged code | 5.8 |  |
| custom classifier on 3.0 (3 classes), tf | 45.9 | 84.3 | 253 | +84 % | 3.3 | 4.0 |
| the same, `library="litert"` | 26.2 | 96.0 | 288 | +266 % | 1.2 | 1.6 |
| custom classifier on 2.4 (3 classes), tf | 242.1 | 268.2 | 805 | +11 % | 2.9 | 3.3 |

Memory is the unique set of all processes of the run together.

Reading it:

- **1.5.1 and main are level** on every leg measured (within 1.1 %): the six
  commits since the release cost nothing.
- **The `onnx` backend is the one to use for 3.0 on a CPU**: 2.1 times `pt`, 10
  times `pb`. `pb` also holds 1.6 GiB resident per worker and takes 20 s to start.
- **TFLite needed XNNPACK.** Without it the 3.0 TFLite file ran at 26 (TensorFlow)
  and 20 (LiteRT) segments per second, against 72 for the same network in ONNX.
  That is the path of every custom classifier trained on 3.0: an export with
  three classes ran at 46 and now runs at 84 (96 on LiteRT), faster
  than the official model because it lacks the 11,725-class heads.
- With XNNPACK, **LiteRT is the better TFLite interpreter**: faster than
  TensorFlow's on both models, a third to two thirds of its memory, and its
  workers are up 1.3 to 2.3 s sooner, without the TensorFlow import.
  `library="tflite"` is still the default of the `tf` backend.
- Perch on a CPU is 33 xRT with 1.5 GiB resident per worker; a worker with 12 threads
  (4.1 seg/s) or two with 6 (5.5) are slower than 12 with one (6.6).

### CPU, worker count (tree)

| workers | 3.0 onnx seg/s | threads per worker | memory GiB | 2.4 tf seg/s | memory GiB |
|---|---|---|---|---|---|
| 1 | 44.2 | 12 | 0.6 | 45.7 | 0.8 |
| 2 | 58.2 | 6 | 1.1 | 87.4 | 0.7 |
| 4 | 70.4 | 3 | 2.1 | 160.0 | 1.3 |
| 6 | 73.2 | 2 | 3.0 | 210.3 | 1.8 |
| 8 | 71.2 | 1 | 3.9 | 237.0 | 2.4 |
| 12 | 71.5 | 1 | 5.8 | 236.7 | 3.6 |
| 16 | 67.2 | 1 | 7.7 | 225.8 | 4.8 |
| 20 | 62.5 | 1 | 9.6 | 214.8 | 6.0 |

- Both models stop scaling at **8 workers, the number of performance cores**.
  The four efficiency cores add nothing and hyper-threads cost 5 to 12 %.
- The 3.0 ONNX model is already level at 4 workers with 3 threads each (2.1 GiB)
  and best at 6 with 2 (3.0 GiB); the default of 12 holds 5.8 GiB for the same
  throughput. It is bound by memory bandwidth, not by cores: one thread alone
  runs 12.4 seg/s, eight together 9.5 each. Whether a smaller default would be
  free elsewhere needs this table from a machine with more memory channels.
- Raw ONNX Runtime, outside the pipeline, reaches 76 seg/s with 12 processes of
  one thread: the pipeline's 72 is within 6 % of what the runtime can do.

### CPU, batch size and producers

| leg | before seg/s | tree seg/s | memory GiB |
|---|---|---|---|
| 3.0 onnx, 12 workers, batch 1 | 70.8 | 70.9 | 5.8 |
| 3.0 onnx, 12 workers, batch 4 | 54.2 | 72.0 | 5.8 |
| 3.0 onnx, 12 workers, batch 16 | 45.8 | 69.7 | 6.0 |
| 3.0 onnx, 1 worker (12 threads), batch 1 | 44.0 | 44.2 | 0.6 |
| 3.0 onnx, 1 worker (12 threads), batch 12 |  | 63.3 | 0.9 |
| 3.0 onnx, 1 worker (12 threads), batch 16 | 58.9 | 67.3 | 0.9 |
| 3.0 onnx, 1 worker (12 threads), batch 24 |  | 67.9 | 0.9 |
| 3.0 onnx, 1 worker (12 threads), batch 48 |  | 73.3 | 0.9 |
| 3.0 onnx, 1 worker (12 threads), batch 64 |  | 74.4 | 0.9 |
| 3.0 onnx, 1 worker (12 threads), batch 128 |  | 74.6 | 1.0 |
| 3.0 onnx, workers left to the library (1), batch 64 |  | 74.9 | 0.9 |
| 3.0 onnx, 12 workers, batch 64 |  | 66.3 | 5.9 |
| 3.0 onnx, 2 workers (6 threads), batch 8 | 63.5 | 68.3 | 1.4 |
| 3.0 onnx, 4 workers (3 threads), batch 4 |  | 70.0 | 2.6 |
| 2.4 tf, 12 workers, batch 1 | 232.1 | 235.0 | 3.6 |
| 2.4 tf, 12 workers, batch 4 | 198.4 | 239.0 | 3.6 |
| 2.4 tf, 12 workers, batch 16 | 168.4 | 234.6 | 3.7 |
| 3.0 onnx, 12 workers, 2 producers | 72.1 | 71.6 | 5.9 |
| 3.0 onnx, 12 workers, 4 producers | 71.4 | 71.7 | 6.0 |
| 2.4 tf, 12 workers, 2 producers | 236.6 | 235.4 | 3.7 |
| 2.4 tf, 12 workers, 4 producers | 234.8 | 234.6 | 3.7 |

"Before" is the tree without the two batch changes (XNNPACK already on).

- On `main` a batch larger than 1 made a CPU run slower, by 24 % at 4 and 36 %
  at 16 for 3.0 ONNX and by 15 % and 28 % for 2.4 TFLite: the activations of a
  batch no longer fit the caches the workers share. The tree runs the rows one
  by one there, so the batch size no longer matters on a CPU.
- One `onnx` worker whose 12 threads each run a row reaches 74 to 75 seg/s at
  batch 64 in 0.9 GiB, against 71 in 5.8 GiB for 12 workers. Its threads idle while
  the last rows of a batch finish, so small batches lose: 63 at 12, 68 at 24,
  73 at 48, 75 at 128. The 12-worker leg at batch 64 (66) is low for another
  reason: a 12 s pass holds only 13 such batches, and the workers end unevenly.
- A second or fourth producer changes nothing on a CPU: the workers are the
  limit (the pipeline's own statistics show 12 of 12 busy and a full buffer).

### CPU, inputs (3.0 onnx unless noted, tree)

| input | seg/s | xRT |
|---|---|---|
| 48 kHz WAV | 70.9 | 213 |
| 32 kHz WAV (the model's rate) | 74.0 | 222 |
| 44.1 kHz WAV | 71.8 | 215 |
| 48 kHz FLAC | 71.6 | 215 |
| 48 kHz MP3 | 71.9 | 216 |
| 9600 files of 15 s | 71.6 | 215 |
| overlap 1.5 s | 71.3 | 107 |
| statistics and progress callback on | 70.8 | 212 |
| embeddings instead of predictions | 71.8 | 215 |
| 2.4 tf, 48 kHz WAV (the model's rate) | 235.0 | 705 |
| 2.4 tf, 32 kHz WAV | 220.2 | 660 |
| 2.4 tf, embeddings (builtin kernels) | 216.5 | 650 |

- At 72 seg/s the input does not matter: 32 kHz (no resampling), 44.1 kHz, FLAC,
  MP3 and 9600 files of 15 s are all within 4 %.
- One producer alone decodes and resamples 1000 seg/s of 48 kHz WAV for the 3.0
  model (4400 without resampling), 300 to 400 of FLAC or MP3. That is the
  ceiling per producer once the workers are faster than that, i.e. on a GPU.
- `show_stats="progress"` with a callback, as an application uses it, costs
  less than 2 %.

### GPU

| leg | main seg/s | tree seg/s | tree xRT | GPU % | device GiB | startup s |
|---|---|---|---|---|---|---|
| 3.0 onnx, 1 worker, batch 1 |  | 261.8 | 785 | 84 | 0.8 | 2.3 |
| 3.0 onnx, 1 worker, batch 16 |  | 575.3 | 1726 | 87 | 1.3 | 2.9 |
| 3.0 onnx, 1 worker, batch 64 | 510.3 | 651.8 | 1956 | 88 | 1.3 | 2.3 |
| 3.0 onnx, 1 worker, batch 64, 2 producers | 483.6 | 648.3 | 1945 | 88 | 1.3 | 2.3 |
| 3.0 onnx, 1 worker, batch 64, 4 producers | 469.6 | 649.0 | 1947 | 88 | 1.3 | 2.7 |
| 3.0 onnx, 1 worker, batch 128 |  | 784.9 | 2355 | 87 | 2.3 | 2.5 |
| 3.0 onnx, 1 worker, batch 128, 4 producers | 500.7 | 774.6 | 2324 | 85 | 2.3 | 2.8 |
| 3.0 onnx, 2 workers, batch 128, 2 producers |  | 856.5 | 2570 | 99 | 4.6 | 2.2 |
| 3.0 onnx fp16, 1 worker, batch 64 |  | 642.4 | 1927 | 87 | 2.3 | 3.9 |
| 3.0 pt, 1 worker, batch 64 |  | 689.0 | 2067 | 86 | 1.6 | 4.0 |
| 3.0 pb, 1 worker, batch 1 | 137.3 | 161.7 | 485 | 40 | 2.8 | 14.3 |
| 2.4 pb, 1 worker, batch 64, 4 producers | 1388.6 | 1485.5 | 4456 | 63 | 3.3 | 5.5 |
| 2.4 pb, 1 worker, batch 256, 4 producers | 1506.4 | 1506.6 | 4520 | 58 | 8.4 | 5.6 |
| Perch v2 pb, 1 worker, batch 64 | 623.9 | 628.9 | 3145 | 57 | 8.4 | 48.6 |

An empty cell was not measured on that side.

- **Best found for 3.0: `onnx`, two workers, batch size 128: 857 seg/s (2570
  xRT)**, 4.6 GiB of device memory. One worker at 128 reaches 785, at 64: 652.
- **The main column of the `onnx` rows is a per-run cost, not a lower speed.**
  main loses about 2.5 s per run to the short last batches (see the changes
  below): 510 seg/s in 10 s passes, 622 in 45 s passes, and a run of hours
  loses nothing measurable. The tree runs 652 and 655.
- ONNX Runtime alone runs 755 seg/s at batch 64 and 937 at 128 on this card; the
  rest is what the single worker does between two batches, with the GPU idle.
  A second worker fills that gap (GPU at 99 %).
- Producers do not limit: 1, 2 and 4 give the same.
- The `fp16` ONNX file is no faster than `fp32` here (642 against 652).
- `pt` is level with `onnx` (689 at batch 64), and needs 11 to 17 s for its
  first run.
- 2.4 `pb` reaches 1507 seg/s (4520 xRT) at batch 256 with the GPU at 58 %: the
  CPU side limits there. Perch: 629 seg/s (3145 xRT), 49 s to start.
- The 3.0 `pb` SavedModel has a static batch of one and reaches 162 seg/s
  with one worker.

How many workers a GPU should get depends on the backend. At batch size 1, a
worker per CPU core (12, the default of `n_workers=None`) against one worker:

| backend | 12 workers seg/s | device GiB | 1 worker seg/s | device GiB |
|---|---|---|---|---|
| 3.0 onnx | 275.7 | 9.4 | 261.8 | 0.8 |
| 3.0 pt | 284.8 | 7.9 | 230.1 | 0.7 |
| 3.0 pb | 337.9 | 11.6 | 137.3 | 2.8 |
| 2.4 pb | 593.2 | 11.5 | 135.2 | 1.1 |

At batch size 64 the 12 `onnx` workers ran out of device memory. The default
does not know the device's memory, and nothing was changed about it: on a GPU,
pass `n_workers`.
- The `pb` and Perch code did not change: where their two columns differ (up to
  7 %), that is the run-to-run spread of these CPU-bound legs.

### What changed in the tree because of these measurements

| change | measured effect here |
|---|---|
| The `tf` backend uses the interpreter's XNNPACK delegate for the acoustic `fp32` models, the 2.4 `fp16` model and custom classifiers, wherever only graph outputs are read | 3.0: 26 → 54 seg/s (TensorFlow's interpreter), 20 → 62 (LiteRT); a custom classifier on 3.0: 46 → 84 and 26 → 96; 2.4: 219 → 235 and 186 → 245. Single thread: 1.2 to 3.4 times. Scores move by float noise (≤ 6e-5 on a 2.4 logit, ≤ 2e-5 on a 3.0 score). Memory per worker: +60 MiB (2.4), +300 MiB (3.0); see the memory column of the first table |
| not for the 2.4 `int8` file | XNNPACK changes its logits by 1.2 to 1.8 |
| not for the 3.0 `fp16` files | the official one is marked `reduced_precision_support = fp16accfp32`, which lets XNNPACK compute in half precision on CPUs that have it (ARMv8.2, AVX512-FP16); not measurable here, where it differs by 4e-6 |
| not for embeddings of 2.4 | they are read from inside the graph, which the delegate does not keep (`Tensor data is null`) |
| The `onnx` backend on a GPU fills a short batch up to the size of the largest one before it | The CUDA provider sets its convolutions up again at every change of the input shape, 1.25 s per change for 3.0 (2.6 s for the fp16 file), also back to a shape it has seen. A run ends with one short batch per producer, so every run lost 2 to 2.5 s: a fifth of a 10 s run (510 → 652 seg/s), 5 % of a 45 s run (622 → 655), nothing of a long one |
| On a CPU the acoustic `tf` backends, and `onnx` with one thread per worker, run the rows of a batch one by one | A batch gains nothing on one thread and costs the caches the workers share: see the batch table |
| An `onnx` worker with several CPU threads runs the rows of a batch in parallel, once the batch has a row for every thread | One worker, 12 threads: 59 → 67 seg/s at batch 16, 74 at batch 64, in 0.9 GiB; 12 workers reach 71 in 5.8 GiB. Raw: one session with one thread per call, called from 12 threads, 79 seg/s, against 61 for 12 threads inside the operators |
| `n_workers=None` starts that one worker for `onnx` on a CPU from four rows per core on | One worker: 63 seg/s at batch 12, 68 at 24, 73 at 48, 74 at 64, 75 at 128; the threads idle while the last rows of a batch finish, so it passes the 71 of a worker per core between 24 and 48 |
| A session validates `device` when it is created; with `n_workers=None` a device list starts one worker per entry | `"gpu"`, `"cuda"` and a GPU for the `tf` backend used to fail in a worker with an `AssertionError` without text, a device list shorter than `n_workers` with an `IndexError`; a longer list, whose extra entries were ignored, is rejected now too |
| `pb` on a GPU that TensorFlow does not see raises `No GPU found! ..` | used to be an `AssertionError` without text |
| `birdnet-benchmark`: `--model-version 3.0`, the `onnx` and `pt` backends, and `--use-perch` runs | Perch was loaded and never run; anything but 2.4 `tf`/`pb` was an `AssertionError` |
| tflite worker memory estimate 180 → 235 MiB plus the model | measured 304 MiB (2.4) and 579 MiB (3.0) per worker with XNNPACK |

#180 (the GPU and device fixes, `birdnet-benchmark`, this harness) and #181
(XNNPACK and batches on a CPU) are on `main`. #182 (parallel rows in one `onnx`
worker) is open: the two table rows above about an `onnx` worker with several
threads, and the rows of the batch table for one worker, describe its branch.
Its first version also started one worker by itself for a large batch; the
rows for workers left to the library at batch size 64 were measured with that
and no longer describe the branch. #184 (the memory plan of the default worker
count) and #185 (no XNNPACK on Apple silicon) are open and come from the two
laptops' sections. Verified on Linux x86-64; the sections of the other
machines hold what they measured.
XNNPACK was checked on TensorFlow
2.20 and 2.21 and on LiteRT 2.0.2 and 2.4; TensorFlow 2.12 and 2.16 do not apply
it to these files and run as before.

### Not done, with the numbers that argue for it

- **One process, many inference threads, for `onnx` on a CPU by default.** One
  ONNX Runtime session called from 12 threads runs 79 to 81 seg/s in 0.7 GiB;
  12 worker processes run 72 to 76 in 5.9 GiB and load the model 12 times. The
  tree gets most of it when asked: `n_workers=1` with `batch_size=64` on the
  branch of #182 runs 73 to 74 seg/s in 0.9 GiB, and loses a little at every
  batch boundary, where the threads wait for the slowest row. Making it the default, and closing that
  gap, means a worker that takes several ring slots at a time, which touches
  the pipeline's crash handling and was left alone. An application can choose
  it today for the 3.0 model.
- **The single `onnx` worker beyond 12 cores.** Its 12 threads on one session
  are measured; 32 or 64 are not, and neither is `n_workers=1` on a machine
  where 12 processes still scale. Compare `3.0-onnx-cpu-wauto-b64-p1` with
  `3.0-onnx-cpu-w<cores>-b1-p1` (both in `cpu-batch` and `cpu-workers`).
- **A default worker count for a GPU.** A worker per CPU core runs out of device
  memory with a batch, and one worker is 2.5 to 4.4 times slower for the `pb`
  backends at batch size 1 (table in the GPU section). A good default needs the
  device's free memory and a per-backend cost, as the CPU side has.
- **XNNPACK for the 3.0 `fp16` TFLite files.** Excluded until someone measures
  on a CPU with half-precision arithmetic (Apple silicon): run
  `3.0-tf-fp16-*` legs with and without it and compare the digests.
- **A smaller default worker count where memory bandwidth limits.** See the
  worker table: 6 workers with 2 threads equal 12 with one for 3.0 ONNX here.
- **LiteRT as the default interpreter of the `tf` backend.** Faster with
  XNNPACK, less memory, no TensorFlow import. An application can already pass
  `library="litert"`.
- **Post-processing off the inference thread on a GPU.** A single GPU worker
  idles the device while it selects the top scores (5.8 ms per batch of 64 for
  3.0, 7.6 ms with the numpy sigmoid for 2.4). Two workers recover it today
  (+8 to +10 %) at twice the device memory.
- **Producers for compressed input.** One producer feeds 300 to 400 seg/s of
  FLAC or MP3. `n_producers` stays 1 unless the caller raises it; nothing
  detects a starved GPU.
- **`pt` on a CPU** runs one thread as fast as ONNX Runtime (96 ms against 81 ms
  per segment) but 12 workers reach only 34 seg/s against 72. Not looked into.
- **TensorRT** for `onnx` on NVIDIA cards, and anything on DirectML: unmeasured.
- **Session logs.** Every session leaves one or two `birdnet_session_*.log`
  files in the temp directory and nothing removes them; 6400 had gathered in
  `/tmp` on this machine. Small files, but an application that opens a session
  per analysis adds two per run.

## win-laptop: Windows 11 laptop, Core Ultra 7 155U (2 P + 8 E + 2 low-power E cores, 14 threads), 31.5 GiB, no NVIDIA GPU

Measured 2026-11-24, Python 3.13.7, onnxruntime 1.32.0, TensorFlow 2.21.0, LiteRT
2.4.0, numpy 2.5.5, scipy 1.18.1, libsndfile 1.2.2: the runtime versions of the
magpie run. No PyTorch (installed with the `tf` extra only), so the `pt` leg
failed on both sides and has no row. On mains, power mode "best performance",
the virus scanner's real-time protection on. `tree` is `onnx-parallel-rows` at
148f940: main with #180 and #181, plus #182. `1.5.1` is the release from PyPI
with the same pins. Corpus and method as for magpie: the fastest of three passes
of about 12 s in a warm session. At 34 seg/s such a pass is two files, 400
segments.

Three things about this machine stand behind every number below:

- **The first legs after a pause are fast.** The same leg (3.0 `onnx`, 12
  workers) ran 38.4 to 39.5 seg/s as the first leg after the machine had rested,
  for one minute or for twenty, 37.1 as the second and 34.3 from the third on.
  That outlasts a leg, so the fastest of three passes does not remove it, and no
  flag shows it. Every number here is from a leg that ran third or later in an
  unbroken row of legs. Seven legs were measured again,
  for that reason or because they were flagged (`noisy`: three, `busy-machine`:
  one); no flag is left.
- **The same setting spreads by 1.6 % either way.** 3.0 `onnx` with 12 workers
  ran between 33.3 and 34.4 seg/s in eleven warm legs over three hours. A
  difference under 3 % between two single legs is not resolved here; where it
  mattered, the legs were run several times in alternating order.
- **13 of 31.5 GiB were free**, other applications held the rest.
  `n_workers=None` is sized from free memory, so 3.0 `pb` ran with 9 workers and
  Perch with 8 instead of 12, on both sides of the comparison. Every other
  default leg got its 12.

### CPU, defaults (one worker per physical core, one thread each, batch size 1)

| model, backend | 1.5.1 seg/s | tree seg/s | tree xRT | change | startup s | memory GiB |
|---|---|---|---|---|---|---|
| 3.0 onnx | 34.2 | 34.3 | 103 | same | 4.2 | 4.8 |
| 3.0 tf (TensorFlow Lite) | 8.3 | 27.7 | 83 | +234 % | 10.7 | 7.2 → 10.6 |
| 3.0 tf, `library="litert"` | 8.1 | 24.6 | 74 | +204 % | 3.6 | 0.8 → 4.5 |
| 3.0 pb (9 workers) | 4.5 | 4.5 | 14 | unchanged code | 21.7 | 10.6 |
| 2.4 tf (TensorFlow Lite) | 73.9 | 140.8 | 422 | +91 % | 10.1 | 3.4 → 4.1 |
| 2.4 tf, `library="litert"` | 73.2 | 123.8 | 372 | +69 % | 2.3 | 0.6 → 1.3 |
| 2.4 pb | 11.3 | 11.3 | 34 | same | 23.3 | 7.9 |
| Perch v2 pb (5 s segments, 8 workers) | 3.3 | 3.2 | 16 | unchanged code | 15.6 | 9.5 |

Memory is the unique set of all processes of the run together; where XNNPACK
changed it, 1.5.1 → tree.

Reading it:

- **1.5.1 and the tree are level where the code is the same.** 3.0 `onnx` in
  alternating order: 34.4 and 34.3 (1.5.1) against 34.1 and 34.2 (tree). Perch:
  a pass of 73.1 s against 74.0 s; `compare` printed -3.0 % there, because it
  divided seg/s rounded to one decimal.
- **XNNPACK gains more here than on Linux.** 3.0: 3.3 times with TensorFlow's
  interpreter, 3.0 times with LiteRT (Linux: 2.1 and 3.1). 2.4: +91 % and +69 %
  (Linux: +7 % and +32 %), in alternating order 73.1 and 73.5 against 140.7 and
  139.7. The builtin kernels are the slow part on this platform: measured against
  3.0 `onnx` on the same machine, 2.4 TFLite without the delegate runs at 2.2
  times its rate here and 3.0 times on Linux, with it at 4.1 and 3.3 times.
- **Embeddings of 2.4 are still on the builtin kernels** and run 73.5 seg/s, half
  of what predictions reach (140.8). On Linux that gap was 8 %.
- **The interpreters swap places.** With XNNPACK TensorFlow's is the faster one
  here (27.7 against 24.6 for 3.0, 140.8 against 123.8 for 2.4), LiteRT's on
  Linux. LiteRT still holds less than half the memory and its workers are up 7 to
  8 s sooner.
- **XNNPACK costs the same memory as on Linux**: +60 MiB per worker for 2.4, +295
  (TensorFlow) and +320 MiB (LiteRT) for 3.0.
- `onnx` stays the backend for 3.0 on a CPU: 34.3 against 27.7 (`tf`) and 4.5
  (`pb`, 9 workers).
- **Digests agree.** 3.0 `onnx` is identical on both sides (430 detections, score
  sum 226.73011). XNNPACK moves the sum of the `tf` legs by 4e-5 (TensorFlow) and
  2e-5 (LiteRT) for 3.0 and by under 4e-6 for 2.4. The `tf` and `pb` sums are
  4e-4 below `onnx` with and without the delegate: that is the export.
- **Startup** (one call minus one pass) is 2.6 to 3.8 times the Linux figure for
  every leg but 3.0 `pb`: 4.2 s against 1.6 for 12 `onnx` workers, 10 to 11 s
  against 3.0 with the TensorFlow import, 23 s against 8.4 for 2.4 `pb`. Not
  broken down into process start, imports and the virus scanner. Opening a
  session with 12 `onnx` workers takes 2.4 s, with one 0.2 s.

### CPU, worker count (tree)

| workers | 3.0 onnx seg/s | threads per worker | memory GiB | 2.4 tf seg/s | memory GiB |
|---|---|---|---|---|---|
| 1 | 20.7 | 12 | 0.5 | 35.6 | 0.4 |
| 2 | 27.2 | 6 | 0.9 | 54.0 | 0.7 |
| 4 | 30.6 | 3 | 1.7 | 78.5 | 1.4 |
| 6 | 32.7 | 2 | 2.4 | 96.2 | 2.1 |
| 8 | 29.4 | 1 | 3.2 | 121.9 | 2.7 |
| 12 | 33.5 | 1 | 4.8 | 140.6 | 4.1 |
| 14 | 33.9 | 1 | 5.6 | 139.4 | 4.7 |

- The 3.0 ONNX model is level from 6 workers with 2 threads each (32.7 seg/s,
  2.4 GiB) to 14 (33.9): the default of 12 holds twice the memory for 2 % more,
  as on magpie. 8 workers are below 6 because they leave four cores idle (one
  thread each, CPU at 62 %).
- 2.4 TFLite scales to 12 workers and gains nothing from the two hyper-threads.
  One worker alone runs 35.6 seg/s, twelve run 11.7 each: ten of the twelve
  cores are efficiency cores.

### CPU, batch size and producers (tree)

| leg | seg/s | memory GiB |
|---|---|---|
| 3.0 onnx, 12 workers, batch 1 | 34.3 | 4.8 |
| 3.0 onnx, 12 workers, batch 4 | 34.1 | 4.8 |
| 3.0 onnx, 12 workers, batch 16 | 31.3 | 4.8 |
| 3.0 onnx, 12 workers, batch 64 | 24.8 | 4.8 |
| 3.0 onnx, 1 worker (12 threads), batch 1 | 20.7 | 0.5 |
| 3.0 onnx, 1 worker (12 threads), batch 16 | 29.6 | 0.6 |
| 3.0 onnx, 1 worker (12 threads), batch 64 | 32.5 | 0.7 |
| 3.0 onnx, workers left to the library (1), batch 64 | 32.5 | 0.7 |
| 3.0 onnx, 2 workers (6 threads), batch 8 | 30.1 | 1.0 |
| 2.4 tf, 12 workers, batch 1 | 140.8 | 4.1 |
| 2.4 tf, 12 workers, batch 4 | 135.2 | 4.4 |
| 2.4 tf, 12 workers, batch 16 | 134.9 | 4.1 |
| 3.0 onnx, 12 workers, 2 producers | 34.1 | 4.9 |
| 3.0 onnx, 12 workers, 4 producers | 34.0 | 5.0 |
| 2.4 tf, 12 workers, 2 producers | 139.1 | 4.1 |
| 2.4 tf, 12 workers, 4 producers | 129.9 | 4.1 |

- A batch of 4 costs 3.0 ONNX nothing; 2.4 TFLite is 4 % lower at 4 and 16. The
  12-worker ONNX legs at 16 (CPU at 73 %) and 64 (46 %) are low because of the
  pass: 400 segments are 25 batches of 16, or 7 of 64, for 12 workers. Not
  measured with a longer pass.
- Producers do not limit: every leg has the CPU at 85 % (12 of 14 threads). Four
  producers cost 2.4 TFLite 8 %, since they take cores from 12 workers on 12
  cores.

### One `onnx` worker with a large batch against a worker per core (#182)

| | 1 worker, batch 64 (`wauto-b64`) | 12 workers, batch 1 (`w12-b1`) |
|---|---|---|
| suite, 12 s passes | 32.5 seg/s | 33.5 seg/s |
| three more of each, alternating | 31.5, 32.8, 32.3 | 33.3, 34.0, 33.8 |
| 48 s passes, two of each, alternating | 33.1, 33.3 | 34.2, 34.4 |
| memory, unique set of all processes | 0.7 GiB | 4.8 GiB |
| startup s | 2.3 | 4.1 |
| CPU % | 80 | 85 |

Here the single worker is 3 to 4 % slower, where magpie had it 3 % faster: -4 %
in 12 s passes (32.3 against 33.7 on average), -3 % in 48 s passes, and in no
pairing ahead. It runs in a seventh of the memory and a call starts 2 s sooner.
`n_workers=None` picked the one worker at batch 64 as intended. Two of these
legs were flagged `busy-machine` (33.3 at 12 workers, 33.3 at one worker with
48 s passes); both sit inside the others. Batch 48, where the rule starts on 12
cores, was not measured.

### CPU, inputs (3.0 onnx unless noted, tree)

| input | seg/s | xRT |
|---|---|---|
| 48 kHz WAV | 34.3 | 103 |
| 32 kHz WAV (the model's rate) | 35.0 | 105 |
| 44.1 kHz WAV | 34.3 | 103 |
| 48 kHz FLAC | 34.3 | 103 |
| 48 kHz MP3 | 34.3 | 103 |
| 9600 files of 15 s | 34.4 | 103 |
| overlap 1.5 s | 34.2 | 51 |
| statistics and progress callback on | 33.4 | 100 |
| embeddings instead of predictions | 34.1 | 102 |
| 2.4 tf, 48 kHz WAV (the model's rate) | 140.8 | 422 |
| 2.4 tf, 32 kHz WAV | 131.8 | 395 |
| 2.4 tf, embeddings (builtin kernels) | 73.5 | 221 |

As on Linux the input does not matter at this rate: everything is within 3 % for
3.0, and resampling costs 2.4 TFLite 6 %.

### Memory per worker against `memory.py`

Unique set per worker in MiB: the slope over 4 to 14 workers where the worker
suite gives one, otherwise the total of the default leg less 70 MiB for the
parent and the producer, divided by its workers. The magpie column is the
largest worker's anonymous resident set, from the comment in `memory.py`.

| backend, model | estimate | here | magpie |
|---|---|---|---|
| onnx, 3.0 | 512 | 400 | 544 |
| tflite with XNNPACK, 2.4 | 294 | 340 | 304 |
| tflite with XNNPACK, 3.0 | 629 | 900 | 579 |
| litert with XNNPACK, 2.4 | 119 | 100 | 99 |
| litert with XNNPACK, 3.0 | 454 | 380 | 374 |
| pb, 2.4 | 684 | 670 | ~700 |
| pb, 3.0 (333 MiB model, 9 workers) | 1349 | 1190 | |
| pb, Perch v2 (8 workers) | 1490 | 1210 | |

The estimates hold or are on the safe side, except for TFLite in TensorFlow's
interpreter: 16 % over for 2.4 and 43 % over for 3.0, where 12 workers hold
10.6 GiB and the plan counts 7.4. Without XNNPACK that worker already holds
606 MiB (1.5.1), where a LiteRT worker holds 60. That is about the 328 MiB of
the model file more than magpie's numbers give for Linux (579 less the 300 of
XNNPACK), as if this interpreter kept a private copy of the file on Windows. Not
looked into; #184 plans with it. These are unique sets, not the commit charge.

### What the recipe needs on this platform

- It runs as written in Git Bash (`venvs/<name>/Scripts/python.exe`), with the
  default model cache and `BIRDNET_OFFLINE=1`, so that a missing model fails its
  leg instead of downloading. 1.5.1 and the tree read the same model files.
- Windows Search indexes every new venv for seven to nine minutes at 7 % of the
  machine, half of the `busy-machine` threshold. Build the venvs and the corpora
  first and start when the CPU is quiet.
- Run two throwaway legs before a suite and after every pause (see above), and
  compare two settings in alternating order.
- Close what holds memory, or pass `--workers`: the default worker count of the
  `pb` and Perch legs follows the free memory.

### Not measured here

- `pt` (PyTorch not installed), and `pb` and Perch at 12 workers.
- A GPU: `nvidia-smi` finds none, and DirectML needs the `onnxruntime-directml`
  package.
- The 12-worker ONNX legs at batch 16 and 64 with a longer pass, and the single
  worker at batch 48.
- The library's test lanes (step 5 of the checklist) were not run on this
  machine.

## m1-air: macOS 15.1, MacBook Air M1 (4 P + 4 E cores), 8 GiB, no fan

Measured 2026-11-24 with magpie's runtimes (Python 3.13.14, onnxruntime 1.32.0,
TensorFlow 2.21.0, LiteRT 2.4.0, numpy 2.5.5, scipy 1.18.1, libsndfile 1.2.2),
corpus and method. `tree` is 148f940: main with #180 and #181, plus #182.
`1.5.1` is the release from PyPI with the same pins. No PyTorch in the venvs, so
the `3.0-pt` leg failed on both sides, and no GPU legs. The 3.0 model is slow
enough here that a pass is the smallest the harness takes (two files): 17 to
32 s for its onnx and tflite legs, 90 to 120 s for 3.0 pb and Perch.

Three things decide how to read this machine's numbers:

- **It heats up.** The first leg after a pause, 40 s are enough, runs 10 to
  15 % faster than the same leg inside a suite, and its passes slow down one
  after the other (16.7, 17.7, 18.2 s). seg/s is the fastest pass, so a cool
  first pass sets it. Two legs compare only when they ran next to each other,
  in both orders.
- **Free memory sets the default worker count, not the cores.** An editor, a
  browser and Docker stayed open: 2.9 to 4.5 GiB were available, 4.5 to 5.6 GiB
  of swap in use, the idle CPU at 6 to 11 %. `n_workers=None` started 5 or 6
  workers for 3.0 onnx, 5 for 3.0 tf, 5 or 7 for 3.0 litert, 1 or 3 for 3.0 pb,
  8 for 2.4 tf, by what was free when the leg began. A `wauto` leg measures that
  fit as much as the code.
- **The *mem MiB* column held the parent process alone in this run** (20 to
  40 MiB); the harness prints the resident sum there now.
  macOS gives the unique set of another process only with its task port, and
  psutil gets `AccessDenied`. Memory below is the summed resident set from the
  result files, which counts a model file once per worker that maps it, and the
  physical footprint (`proc_pid_rusage`: private pages, compressed and swapped
  ones included, what Activity Monitor shows), sampled from outside in runs of
  their own.

### CPU, defaults (workers left to the library, batch size 1)

| model, backend | 1.5.1 seg/s (workers) | tree seg/s (workers) | tree xRT | startup s | resident GiB |
|---|---|---|---|---|---|
| 3.0 onnx | 18.5 (5) | 19.9 (5) | 60 | 2.2 | 2.6 |
| 3.0 tf (TensorFlow Lite) | 15.7 (5) | 17.0 (5) | 51 | 3.9 | 4.5 |
| 3.0 tf, `library="litert"` | 15.9 (5) | 20.3 (7) | 61 | 1.7 | 5.0 |
| 3.0 pb | 3.3 (1) | 3.9 (3) | 12 | 15.1 | 3.0 |
| 2.4 tf (TensorFlow Lite) | 97.7 (8) | 101.4 (8) | 304 | 5.4 | 4.1 |
| 2.4 tf, `library="litert"` | 98.7 (8) | 99.8 (8) | 299 | 0.6 | 1.7 |
| 2.4 pb | 61.7 (3) | 66.7 (4) | 200 | 11.1 | 2.8 |
| Perch v2 pb (5 s segments) | 2.4 (1) | 2.7 (2) | 13 | 7.5 | 2.1 |

Where the two sides started different numbers of workers, that is what the
two columns compare; the next table holds the count fixed.

### XNNPACK, four workers on both sides

| leg, batch size 1 | 1.5.1 seg/s | tree seg/s | footprint per worker, 1.5.1 → tree |
|---|---|---|---|
| 3.0 tf (TensorFlow Lite) | 14.8 to 17.3 (4 runs) | 15.4 to 16.7 (5 runs) | 310 → 584 MiB |
| 3.0 tf, `library="litert"` | 15.0, 15.3 | 15.9, 16.0 | 117 → 391 MiB |
| 2.4 tf (TensorFlow Lite) | 78.5, 81.0 | 80.4, 81.4, 82.6 | 255 → 303 MiB |
| 2.4 tf, `library="litert"` | 80.7, 83.3 | 80.2, 83.0, 85.3 | 57 → 109 MiB |

- **XNNPACK gains next to nothing on this CPU**: within 2 % for 2.4, about 5 %
  for 3.0, where x86-64 gained a factor of 2.1 to 3.1. The 3.0 TensorFlow leg
  moves more between two runs of one side than between the sides.
- **It costs the same memory as on magpie**: +274 MiB per worker for 3.0, +50
  for 2.4. So the delegate is applied; the builtin kernels are just not slow
  on this CPU. #185 leaves it off on Apple silicon for that.
- Scores: the 3.0 score sum over 430 detections moves by 3e-5 (TensorFlow) and
  1e-5 (LiteRT) against the builtin kernels, the 2.4 sum by 2e-5; detections
  and species are the same on every backend and both versions. The 3.0 TFLite
  file differs from the ONNX file by 3.4e-4 in that sum without the delegate
  and by 3.7e-4 with it.
- LiteRT and TensorFlow's interpreter are equally fast here. LiteRT holds
  190 MiB less per worker and its workers are up sooner (0.6 against 5.4 s for
  eight 2.4 workers).

### One `onnx` worker with a large batch against a worker per core (#182)

`3.0-onnx-cpu-wauto-b64-p1` (the library starts one worker) and
`3.0-onnx-cpu-w8-b1-p1`, four rounds, the order swapped each round:

| round | ran first | one worker, batch 64 seg/s | eight workers, batch 1 seg/s |
|---|---|---|---|
| 1 | one worker, after a pause | 24.0 | 21.2 |
| 2 | eight workers | 21.1 | 21.5 |
| 3 | one worker | 21.4 | 20.8 |
| 4 | eight workers | 20.9 | 21.2 |

| | one worker, batch 64 | eight workers, batch 1 |
|---|---|---|
| seg/s, rounds 2 to 4 | 21.1 | 21.2 |
| one whole call of 400 segments | 19.8 to 20.3 s | 22.2 to 22.4 s |
| footprint of the run | 0.35 to 0.41 GiB | 3.3 GiB (three runs within 6 MiB) |
| resident set of the run | 1.1 to 1.2 GiB | 3.6 to 3.8 GiB |
| swap while it ran | unchanged or falling | grew by 0.1 to 0.9 GiB |

- **It holds here: the same throughput in an eighth of the footprint**, and a
  call is back 2 s sooner because one model loads instead of eight. Not the 3 %
  gain of magpie, and no loss. The 10 % lead in the suite (22.8 against 20.7)
  was the order of the legs.
- One worker is level from batch 16 on (21.1, 20.9 at 64, measured one after the
  other), so starting it from 32 rows on (four per core, the efficiency cores
  counted) is on the safe side here.
- Without a batch size the default is 5 or 6 workers at 17.5 to 19.9 seg/s in
  2.4 to 2.9 GiB resident; `batch_size=64` gives 21 in 1.2 GiB.

### CPU, worker count (tree)

| workers | 3.0 onnx seg/s | resident GiB | 2.4 tf seg/s | resident GiB |
|---|---|---|---|---|
| 1 | 12.6 | 0.8 | 32.9 | 0.6 |
| 2 | 16.4 | 1.5 | 57.3 | 1.2 |
| 4 | 19.1 | 2.4 | 81.4 | 2.2 |
| 8 | 20.7 | 3.6 | 104.4 (114.8 as the first leg after a pause) | 4.4 |

- Four workers more than there are performance cores add 8 % for 3.0 onnx and
  28 % for 2.4 tf.
- Nothing stops scaling before the eight cores are used, but 3.0 onnx gains
  only 26 % from the second to the eighth worker.

### CPU, batch size, producers and inputs (tree)

| leg | workers | seg/s |
|---|---|---|
| 3.0 onnx, workers left to the library, batch 4 | 5 | 21.9 |
| 3.0 onnx, workers left to the library, batch 16 | 4 | 22.4 |
| 3.0 onnx, workers left to the library, batch 64 | 1 | 22.8 |
| 3.0 onnx, 1 worker, batch 1 | 1 | 12.6 |
| 3.0 onnx, 1 worker, batch 16 | 1 | 21.1 |
| 3.0 onnx, 1 worker, batch 64 | 1 | 20.9 |
| 3.0 onnx, 2 workers, batch 8 | 2 | 20.9 |
| 3.0 onnx, 8 workers, batch 64 | 8 | 18.8 |
| 2.4 tf, workers left to the library, batch 4 | 8 | 104.6 |
| 2.4 tf, workers left to the library, batch 16 | 5 | 82.9 |
| 3.0 onnx, 2 producers / 4 producers | 6 | 19.0 / 19.0 |
| 2.4 tf, 2 producers / 4 producers | 8 | 102.2 / 100.4 |
| 3.0 onnx, 32 kHz WAV / 44.1 kHz WAV | 6 / 5 | 18.8 / 17.5 |
| 3.0 onnx, 48 kHz FLAC / MP3 | 6 / 5 | 19.0 / 17.4 |
| 3.0 onnx, 9600 files of 15 s | 5 | 17.5 |
| 3.0 onnx, overlap 1.5 s (26 xRT) | 5 | 17.5 |
| 3.0 onnx, statistics and progress callback on | 6 | 17.5 |
| 3.0 onnx, embeddings | 5 | 17.7 |
| 2.4 tf, 32 kHz WAV | 8 | 97.7 |
| 2.4 tf, embeddings (builtin kernels) | 8 | 99.3 |

- Producers and inputs change nothing that the worker count (5 or 6) and the
  order of the legs do not explain; the first three rows ran first and cool.
- **The default lost three workers and 20 % at batch 16 for 2.4 tf.** The
  estimate adds 15 MiB per row of the batch, and the rows run one by one now:
  see the next table.

### Memory per worker against `memory.py`

Peak footprint of a worker (the kernel's lifetime maximum), four workers unless
noted, against `WorkerCost.estimate`:

| worker | measured MiB | estimate MiB |
|---|---|---|
| tflite 2.4 | 300 to 308 | 309 |
| tflite 3.0 | 580 to 584 | 644 |
| litert 2.4 | 107 to 112 | 134 |
| litert 3.0 | 389 to 392 | 469 |
| onnx 3.0, 4 or 8 workers | 368 to 438 | 524 |
| tflite 2.4, batch 16 | 372 to 374 | 534 |
| onnx 3.0, batch 16 | 375 to 468 | 704 |
| onnx 3.0, one worker, batch 64 | 387 to 392 | 1280 |

- At batch size 1 the estimates are 0 to 20 % above the largest worker
  measured: safe here.
- A row of the batch costs 4.5 MiB (tflite 2.4) where the estimate adds 15, and
  the one `onnx` worker at batch 64 peaks where a worker at batch 1 does. On a
  machine with memory to spare that is invisible; here it sets the worker count.
  Read afterwards from the samples of the batch-16 run: each worker grew by
  60 to 72 MiB over the first 15 s, and the audio ring of that run is 70 MiB
  (8 slots of 16 segments at 0.55 MiB). Every worker maps the ring, so that
  growth is what a footprint shows if it counts shared memory in each process
  that touches it. On magpie the resident set of a worker grows the same way
  and its anonymous memory does not (306 MiB at batch sizes 1 and 16). So the
  4.5 MiB are most likely the ring, once, and not the workers' own; not
  checked on a Mac. #184 plans such a batch at 2 MiB per further row.
- The parent holds 38 to 41 MiB, a producer 48 to 96 MiB at batch size 1 and
  124 to 157 MiB at 16 and 64.
- Four `onnx` workers held 1.6 GiB through one run and 0.8 GiB through another
  with the same peaks: half of a worker's footprint after loading is given back
  in some runs and kept in others. Eight held 3.3 GiB in all three runs.

### What the recipe needed here

- A Python that is not signed as a debugger. psutil's unique set of a child
  needs the task port: with a uv-managed interpreter the call is refused at once
  and the suite runs. An interpreter with the debugger entitlement, as the
  python.org installers carry, made macOS ask for an administrator password at
  that call on this machine before, and the caller blocks on the dialog. Not
  tried again for this run.
- The editable install and the two venvs outside the checkout, the model cache
  left where it was with `BIRDNET_OFFLINE=1`: no model was downloaded.
- A suite started from an interactive session trips `busy-machine` on its first
  leg (the session's own drawing: 18 to 22 % for a few seconds); a minute's
  pause before the first leg avoids it. Three legs were flagged in all and
  measured again without a flag.

### Not done here

- **The harness's memory column on macOS.** `proc_pid_rusage` reads the
  footprint of a child without the task port; the sampler used here lives
  outside the repository.
- **The worker count of a `wauto` leg** is not in the result file. It is
  `processes_peak` minus the parent, the resource tracker and the producers,
  and on a machine short of memory it is half of what the leg means.
- **`pt`**, the custom classifier legs and 1.5.1 beyond `cpu-backends` and the
  four-worker TFLite legs: the batch and worker tables have no "before" here.
- **XNNPACK for the 3.0 `fp16` TFLite files** on this CPU, which has
  half-precision arithmetic: the library excludes them, so it needs a patched
  tree.
- **The library's test lanes** were not run on this machine; the same commit is
  green on the `macos-15` runners.
- **An idle machine in the strict sense**, a cool start for every leg, and why
  the `onnx` workers keep or return half their footprint.
