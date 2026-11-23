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
     delegate. Linux x86-64 gained a factor of 2.1 to 3.1 for 3.0 and 7 to 32 %
     for 2.4. On ARM the delegate uses other kernels (NEON); expect a gain, and
     check the digest column agrees with the `onnx` leg to the fourth decimal.
     It costs memory: compare the *mem MiB* column of those legs too.
   - **Worker scaling** (`cpu-workers`): where throughput stops growing. Here
     the 3.0 ONNX model stops at 4 to 6 workers of 12 cores, because memory
     bandwidth is used up; a machine with more memory channels per core (Apple
     silicon, servers) should scale further, and hybrid CPUs get little from
     their efficiency cores.
   - **One `onnx` worker with a large batch** (`3.0-onnx-cpu-wauto-b64-p1`)
     against a worker per core (`3.0-onnx-cpu-w<cores>-b1-p1`): here the single
     process is 3 % faster in a sixth of the memory. The library now picks it
     by itself from four rows per core on, so it has to hold on every platform.
   - **startup s**: process start is cheap on Linux. On Windows (spawn, virus
     scanner) and macOS it decides how long a single short file takes.
   - **uss MiB** per worker against the estimates in
     `src/birdnet/acoustic/inference/memory.py`; they were measured on Linux.
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

The changes are proposed in three pull requests: #180 (the GPU and device
fixes, `birdnet-benchmark`, this harness), #181 (XNNPACK and batches on a CPU)
and #182 (parallel rows in one `onnx` worker and its default). Measured and
verified on Linux x86-64 only. XNNPACK was checked on TensorFlow
2.20 and 2.21 and on LiteRT 2.0.2 and 2.4; TensorFlow 2.12 and 2.16 do not apply
it to these files and run as before.

### Not done, with the numbers that argue for it

- **One process, many inference threads, for `onnx` on a CPU by default.** One
  ONNX Runtime session called from 12 threads runs 79 to 81 seg/s in 0.7 GiB;
  12 worker processes run 72 to 76 in 5.9 GiB and load the model 12 times. The
  tree gets most of it when asked: `batch_size=64` (which now defaults to one
  worker) runs 73 to 74 seg/s in 0.9 GiB, and loses a little at every batch boundary, where the
  threads wait for the slowest row. Making it the default, and closing that
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
