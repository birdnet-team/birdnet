Benchmarking using command-line tool
============

This page provides information on benchmarking BirdNET's performance on different hardware configurations. For some real world benchmark results, see :doc:`Comparative benchmarking results <benchmarking_results>`.

.. note::

   The information provided here is not up-to-date with the latest developments in the BirdNET library.

.. contents::
   :local:
   :depth: 2

Example usage
-----------------

- Show benchmark options: ``birdnet-benchmark --help``
- Predict top 5 species for each segment using CPU und TFLite backend (single file): ``birdnet-benchmark soundscape.wav``
- Predict all audio files in a directory: ``birdnet-benchmark path/to/audio/files/``
- Use Protobuf backend: ``birdnet-benchmark soundscape.wav -b "pb"``
- Output predictions for top 10 species: ``birdnet-benchmark soundscape.wav --top-k 10 --confidence -100``
- Run on GPU: ``birdnet-benchmark soundscape.wav --backend "pb" --worker 1 --device "GPU" --batch-size 1000``
  - To determine the largest possible batch size, you must experiment with several values. On a GPU with 24 GB of VRAM, a batch size of roughly 1,000 usually works well. If the batch size is set too high, the pipeline will abort with a runtime error ("Analysis was cancelled due to an error."), and the log will state that the GPU ran out of memory.
- Run on three GPUs: ``birdnet-benchmark soundscape.wav --backend "pb" --worker 3 --device "GPU:0" "GPU:1" "GPU:2" --batch-size 1000``
- Increase amount of *Producers*: ``birdnet-benchmark soundscape.wav --producers 2``
- Increase *Buffer* size to 3 * *Worker*: ``birdnet-benchmark soundscape.wav --prefetch-ratio 2``

File types within the run folder
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Category
     - Filename
     - Contents
   * - **Runtime Statistics**
     - ``stats-{timestamp}.txt``
     - Summary of the key metrics for the run.
   * -
     - ``stats-{timestamp}.json``
     - Complete metric set in JSON format.
   * - **Inference Results**
     - ``result-{timestamp}.npz``
     - Space-efficient binary file containing per-segment probabilities for all species (source for all other formats).
   * -
     - ``result-{timestamp}.csv``
     - Tabular view of probabilities; first column holds the full recording path.
   * - **Log**
     - ``log-{timestamp}.log``
     - Full log of the benchmark run.

**Cross-Run Overview** – The parent directory maintains a file named ``runs.csv`` containing the metrics of **all** runs in chronological order, enabling comparative analyses.
Example output on Linux:

.. code-block:: text

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


Interpretation of runtime metrics
--------------------------------

During analysis, performance indicators are updated and printed once per second.

.. list-table::
   :header-rows: 1
   :widths: 15 50 35

   * - Abbr.
     - Meaning
     - Target / Recommendation
   * - **SPEED**
     - Acceleration factor relative to real-time (RT). ``2 xRT`` means ten minutes of audio are processed in five. Startup overhead and one-time model loading per process are excluded. Derived from mean worker runtime relative to processed audio duration. Also reports segments/sec.
     - As high as possible; typically ≥ 50 xRT.
   * - **MEM**
     - Total main-memory usage of the Python parent process plus subprocesses, including shared memory (MB).
     - Keep below available RAM.
   * - **BUF**
     - Average number of batches in the buffer, shown as ``current / maximum``.
     - For ``W`` workers: ``BUF ≈ 2W / 2W``.
   * - **WAIT**
     - Mean waiting time (ms) that workers spend waiting for new batches.
     - NVMe SSD: ≤ 1 ms.
   * - **BUSY**
     - Average number of simultaneously active workers.
     - Ideally ``W / W``.
   * - **PROG**
     - Overall analysis progress in percent.
     - 0% → 100%.
   * - **ETA**
     - Estimated time to completion.
     - As small as possible.

Example log line::

   SPEED: 51 xRT [17 seg/s]; MEM: 1590 M; BUF: 8/8; WAIT: 0.17 ms; BUSY: 4/4; PROG: 93.5 %; ETA: 0:00:48


Typical bottlenecks and mitigation measures
^^^^

* **High WAIT values or empty buffer** – Increase the number of *Producers*. If insufficient, use faster storage (NVMe/SSD) or reduce *Workers*.
* **BUSY < Worker count** – Typically an I/O bottleneck. Apply steps above.
* **Cache effect** – OS file caching boosts SPEED significantly on the second pass. For benchmarking, use only runs starting from the second pass.


Metrics after completion
-----

After analysis completes, the benchmark tool reports:

* **Total Execution Time (Wall Time)** – Program start → completion.
* **Average Buffer Size (Buffer)** – Mean number of batches in the working buffer.
* **Worker Utilisation (Busy Workers)** – Average number of active workers; mean wait time shown in parentheses.
* **Memory Utilisation (Memory Usage)** – Peak RAM consumption including buffer and result array.
* **Processing Throughput (Performance)** – **Most informative metric.** Expressed as × real-time: cumulative audio hours divided by total execution time. Also shows segments/sec and audio-sec/sec.
* **Computational Performance (Worker Performance)** – Final compute speed, identical to the final SPEED value.

