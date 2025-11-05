# Birdnet Benchmark Command-line Tool

## General Functionality of the Python Library

![Birdnet Structure](./img/birdnet-structure.svg)

The analysis pipeline processes **recordings** with five logically distinct components:

1. **Feeder** – Read the recordings, split them into 3-second segments, group them to batches, and fill the buffer.
2. **Buffer** – An intermediate store that holds batches of 3-second audio segments.
3. **Worker** – Take batches from the buffer and perform inference with the model.
4. **Consumer** – Receives the probabilities calculated by the *Workers* and writes them to the result array.
5. **Result** – A three-dimensional matrix in which
     - **Dimension 1** represents the recordings,
     - **Dimension 2** the consecutive 3-second segments, and
     - **Dimension 3** the species covered by the model.
    - Each matrix cell stores the predicted probability for a given species in the corresponding segment of the file.
  
### Parallelisation and Resource Management

* **Number of Processes** – The numbers of *Feeder* and *Worker* processes are configurable. By default, one (1) *Feeder* is launched, while the number of *Workers* equals the count of *physical* CPU cores in the system. *Feeders* and *Workers* run concurrently: *Feeders* preload batches into the buffer, and *Workers* consume those batches for inference. A *Feeder* loads only as much audio as the buffer can hold, keeping RAM usage low because at any moment only the required 3-second segments are in memory.
* **Buffer Size** – By default, the buffer is set to twice the *Worker* count, ensuring that every *Worker* always has a pre-loaded batch to process and thus avoids idle time.
* **Model Backends** – Each worker loads its own instance of the inference model. On the CPU, both **TFLite** and **Protocol Buffers** (Protobuf) models can be used; Protobuf models can optionally run on the GPU.
* **Best Practice for CPU Inference** – For CPU-only execution on Linux, the number of *Worker* processes should not exceed the number of physical cores, as oversubscription typically leads to reduced performance. When running TFLite, keep the batch size to one (1); larger batches offer no throughput benefit.

## Setup

A Python 3.11 installation is required. If you don't have it, you can install it from [python.org](https://www.python.org/downloads/release/python-3119/).
After Python is installed, you can install the `birdnet` package via the provided wheel file. Get newest `birdnet` version via TUCcloud: [https://tuc.cloud/index.php/s/Qtace7JWnTKAe88](https://tuc.cloud/index.php/s/Qtace7JWnTKAe88)

The installation process creates a virtual Python environment called `.venv-bn` and installs the `birdnet` package into it. This is recommended to avoid conflicts with other Python packages.

### Install on Windows (CMD)

```cmd
py -3.11 -m venv .venv-bn
.venv-bn\\Scripts\\activate
python.exe -m pip install --upgrade pip
python.exe -m pip install wheel
python.exe -m pip install birdnet-0.2.0a0-py3-none-any.whl
```

### Install on Linux (Bash)

```sh
python3.11 -m venv .venv-bn
source .venv-bn/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install birdnet-0.2.0a0-py3-none-any.whl
```

### Install with GPU Support

Use the suffix `[and-cuda]` while installing, i.e., `... pip install birdnet-0.2.0a0-py3-none-any.whl[and-cuda]` to support running the benchmark on a GPU with CUDA support.

### Upgrade Version

Just install the new version in the activated environment.

## Example Usage

- Show benchmark options: `birdnet-benchmark --help`
- Predict top 5 species for each segment using CPU und TFLite backend (single file): `birdnet-benchmark soundscape.wav`
- Predict all audio files in a directory: `birdnet-benchmark path/to/audio/files/`
- Use Protobuf backend: `birdnet-benchmark soundscape.wav -b "pb"`
- Output predictions for top 10 species: `birdnet-benchmark soundscape.wav --top-k 10 --confidence -100`
- Run on GPU: `birdnet-benchmark soundscape.wav --backend "pb" --worker 1 --device "GPU" --batch-size 1000`
  - To determine the largest possible batch size, you must experiment with several values. On a GPU with 24 GB of VRAM, a batch size of roughly 1,000 usually works well. If the batch size is set too high, the pipeline will abort with a runtime error ("Analysis was cancelled due to an error."), and the log will state that the GPU ran out of memory.
- Run on three GPUs: `birdnet-benchmark soundscape.wav --backend "pb" --worker 3 --device "GPU:0" "GPU:1" "GPU:2" --batch-size 1000`
- Increase amount of *Feeders*: `birdnet-benchmark soundscape.wav --feeders 2`
- Increase *Buffer* size to 3 * *Worker*: `birdnet-benchmark soundscape.wav --prefetch-ratio 2`

## Result Files

All BirdNET data is stored in the application-data directory, whose location is platform-specific:

- **Windows:** `%APPDATA%/birdnet`
- **Linux:** `~/.local/share/birdnet`
- **macOS:** `~/Library/Application Support/birdnet`

Benchmark results for each run are stored in a dedicated sub-folder `birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-{timestamp}`.

### File Types within the Run Folder

| Category | Filename | Contents |
|----------|----------|----------|
| **Runtime Statistics** | `stats-{timestamp}.txt` | Summary of the key metrics for the run. |
|  | `stats-{timestamp}.json` | Complete metric set in JSON format. |
| **Inference Results** | `result-{timestamp}.npz` | Space-efficient, fast-savable binary file containing per-segment probabilities for all species—serves as the source for all other formats. |
|  | `result-{timestamp}.csv` | Tabular view of the probabilities; the first column holds the full path of the recording. |
| **Log** | `log-{timestamp}.log` | Full log of the benchmark run. |

**Cross-Run Overview** – The parent directory also maintains a file named `runs.csv`, which contains the metrics of **all** benchmark runs in chronological order and thus enables comparative analyses.

<details>

<summary><b>Example output on Linux</b></summary>

```txt
Benchmark folder:
  /home/user/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-20250710T143348
Statistics results written to:
  /home/user/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-20250710T143348/stats-20250710T143348.txt
  /home/user/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-20250710T143348/stats-20250710T143348.json
  /home/user/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/runs.csv
Prediction results written to:
  /home/user/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-20250710T143348/result-20250710T143348.npz
  /home/user/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-20250710T143348/result-20250710T143348.csv
Log file written to:
  /home/user/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-20250710T143348/log-20250710T143348.log
```
</details>

## Interpretation of Runtime Metrics

During analysis, the key performance indicators are updated and printed once per second.

| Abbr. | Meaning | Target / Recommendation |
|-------|---------|-------------------------|
| **SPEED** | *Acceleration factor* relative to real-time (RT). A value of `2 xRT` means that ten minutes of audio can be processed in five minutes. The calculation excludes programme start-up and the one-time model loading per process. SPEED is derived from the mean runtime of all *Worker* processes in relation to the total duration of audio already processed. The number of 3-second segments per second is also reported. | As high as possible; typically ≥ 50 xRT |
| **MEM** | Total main-memory usage of the Python parent process *plus* all subprocesses, including shared memory (in MB). | Keep below the available RAM capacity |
| **BUF** | Average number of batches in the buffer, shown as *current / maximum*. | For `W` *Workers*: `BUF ≈ 2W / 2W` |
| **WAIT** | Mean waiting time (in ms) that *Workers* spend waiting for a new batch in the buffer. | NVMe SSDs: ≤ 1 ms |
| **BUSY** | Average number of simultaneously busy *Workers*, shown as *active / total*. | Aim for `W / W` |
| **PROG** | Overall analysis progress in %. | Increases linearly from 0 % → 100 % |
| **ETA** | Estimated time remaining until completion. | As small as possible |


```text
Example log line
----------------
SPEED: 51 xRT [17 seg/s]; MEM: 1590 M; BUF: 8/8; WAIT: 0.17 ms; BUSY: 4/4; PROG: 93.5 %; ETA: 0:00:48
```

### Typical Bottlenecks and Mitigation Measures

* **High WAIT values or an empty buffer** – Increase the number of *Feeders*. If that is not sufficient, copy the audio data to faster storage (NVMe/SSD) or reduce the number of *Workers*.
* **BUSY lower than the Worker count** – Usually the same bottleneck as above (I/O constraint). Apply the steps listed above.
* **Cache effect** – Because operating systems cache files in RAM, SPEED often increases markedly on a second pass over the same audio data. For benchmarking, evaluate only runs from the second attempt onwards.

## Interpretation of Metrics After Analysis

After the analysis has completed, the benchmark tool reports the following key figures:

* **Total Execution Time (*Wall Time*)** – The total time in seconds from program start to completion.
* **Average Buffer Size (*Buffer*)** – The mean number of batches simultaneously present in the working buffer.
* **Worker Utilisation (*Busy Workers*)** – The average number of *Workers* active in parallel. The mean waiting time until a new batch became available is shown in parentheses.
* **Memory Utilisation (*Memory Usage*)** – The peak main-memory consumption of the process together with the sizes of the buffer and the result array.
* **Processing Throughput (*Performance*)** – **This is the most informative metric for estimating overall processing speed.** It is expressed as a multiple of real-time, calculated by dividing the cumulative hours of audio processed by the total execution time. The report also lists the mean number of 3-second segments processed per second and the corresponding audio duration handled per second.
* **Computational Performance (*Worker Performance*)** – The final compute speed, identical to the SPEED value after all *Workers* have finished.

## Comparative Results

### Run 10 h WAV-files on Intel i7-8565U 4-Core with 16 GB RAM (Windows 10)

- Input: 10x 60 minute WAV-files
- Disk: NVMe SSD (Intel SSDPEKKF010T8L)
- Device: Lenovo ThinkPad X1 Carbon 7th Gen

<details open>
<summary><b>Result:</b> 49 x real-time (RTF: 0.02047110)</summary>

```sh
$ birdnet-benchmark test-dataset/test_dataset_100x60min -f 5

-------------------------------
------ Benchmark summary ------
-------------------------------
Start time: 07/09/2025 03:47 PM
End time:   07/09/2025 04:00 PM
Wall time:  0:12:16.959720
Input: 10 file(s) (WAV)
  Total duration: 10:00:00
  Average duration: 1:00:00
  Minimum duration (single file): 1:00:00
  Maximum duration (single file): 1:00:00
Feeder(s): 1
Buffer: 8.0/8 filled slots (mean)
Busy workers: 3.9/4 (mean)
  Average wait time for next batch: 0.257 ms
Memory usage:
  Program: 1596.43 M (total max)
  Buffer: 4.39 M (shared memory)
  Result: 0.34 M (NumPy)
Performance:
  49 x real-time (RTF: 0.02047110)
  16 segments/s (0:00:48.849346 audio/s)
Worker performance:
  50 x real-time (RTF: 0.01992776)
```
</details>

### Run 100 h WAV-files on AMD Ryzen 7 3800X 8-Core with 64 GB RAM (Linux)

- Input: 100x 60 minute WAV-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)

<details>
<summary><b>Result:</b> 416 x real-time (RTF: 0.00240288)</summary>

```sh
$ birdnet-benchmark test-dataset/test_dataset_100x60min -f

-------------------------------
------ Benchmark summary ------
-------------------------------
Start time: 07/09/2025 03:15 PM
End time:   07/09/2025 03:30 PM
Wall time:  0:14:25.035450
Input: 100 file(s) (WAV)
  Total duration: 4 days, 4:00:00
  Average duration: 1:00:00
  Minimum duration (single file): 1:00:00
  Maximum duration (single file): 1:00:00
Feeder(s): 1
Buffer: 15.4/16 filled slots (mean)
Busy workers: 8.0/8 (mean)
  Average wait time for next batch: 0.044 ms
Memory usage:
  Program: 1849.19 M (total max)
  Buffer: 8.79 M (shared memory)
  Result: 2.94 M (NumPy)
Performance:
  416 x real-time (RTF: 0.00240288)
  139 segments/s (0:06:56.167916 audio/s)
Worker performance:
  417 x real-time (RTF: 0.00239808)
```
</details>

### Run 100 h WAV-files on NVIDIA Titan RTX with 24 GB (Linux)

- Input: 100x 60 minute WAV-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)

<details>
<summary><b>Result:</b> 2702 x real-time (RTF: 0.00037009)</summary>

- Disk speed: `$ sudo hdparm -tT /dev/nvme0n1`
  - Timing cached reads: 27324 MB in 1.99 seconds = 13707.90 MB/sec
  - Timing buffered disk reads: 6880 MB in 3.00 seconds = 2293.17 MB/sec

```sh
$ birdnet-benchmark test-dataset/test_dataset_100x60min --device GPU --backend pb -w 1 -f 5 -s 1025 --prefetch-ratio 5

-------------------------------
------ Benchmark summary ------
-------------------------------
Start time: 11/05/2025 01:54 PM
End time:   11/05/2025 01:56 PM
Wall time:  0:02:13.231957
Input: 100 file(s) (WAV)
  Total duration: 4 days, 4:00:00
  Average duration: 1:00:00
  Minimum duration (single file): 1:00:00
  Maximum duration (single file): 1:00:00
Feeder(s): 5
Buffer: 1.7/6 filled slots (mean)
Busy workers: 0.9/1 (mean)
  Average wait time for next batch: 0.001 ms
Memory usage:
  Program: 8543.88 M (total max)
  Buffer: 3295.93 M (shared memory)
  Result: 5.58 M (NumPy)
Performance:
  2702 x real-time (RTF: 0.00037009)
  901 segments/s (0:45:02.054441 audio/s)
Worker performance:
  2869 x real-time (RTF: 0.00034853)
  956 segments/s (0:47:49.160292 audio/s)
```
</details>

### Run 100 h FLAC-files on NVIDIA Titan RTX with 24 GB (Linux)

- Input: 100x 60 minute FLAC-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)

<details>
<summary><b>Result:</b> 2487 x real-time (RTF: 0.00040214)</summary>

```sh
$ birdnet-benchmark test-dataset/test_dataset_100x60min_flac --device GPU --backend pb -w 1 -f 5 -s 1025

-------------------------------
------ Benchmark summary ------
-------------------------------
Start time: 07/09/2025 03:07 PM
End time:   07/09/2025 03:09 PM
Wall time:  0:02:24.770857
Input: 100 file(s) (FLAC)
  Total duration: 4 days, 4:00:00
  Average duration: 1:00:00
  Minimum duration (single file): 1:00:00
  Maximum duration (single file): 1:00:00
Feeder(s): 5
Buffer: 1.9/2 filled slots (mean)
Busy workers: 1.0/1 (mean)
  Average wait time for next batch: 0.051 ms
Memory usage:
  Program: 9304.52 M (total max)
  Buffer: 1126.11 M (shared memory)
  Result: 2.94 M (NumPy)
Performance:
  2487 x real-time (RTF: 0.00040214)
  829 segments/s (0:41:26.688317 audio/s)
Worker performance:
  2602 x real-time (RTF: 0.00038431)
```
</details>