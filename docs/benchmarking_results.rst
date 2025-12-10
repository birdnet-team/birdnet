Comparative benchmarking results
===================

.. contents::
   :local:
   :depth: 2

Run 10 h WAV-files on Intel i7-8565U (4-Core, 16 GB RAM, Windows 10)
--------------------------------------------------------------------

- Input: 10 × 60-minute WAV-files
- Disk: NVMe SSD (Intel SSDPEKKF010T8L)
- Device: Lenovo ThinkPad X1 Carbon 7th Gen

**Result:** 49 × real-time (RTF: 0.02047110)

.. code-block:: txt

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


Run 100 h WAV-files on AMD Ryzen 7 3800X (8-Core, 64 GB RAM, Linux)
-------------------------------------------------------------------

- Input: 100 × 60-minute WAV-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)

**Result:** 416 × real-time (RTF: 0.00240288)

.. code-block:: txt

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


Run 100 h WAV-files on NVIDIA Titan RTX (24 GB, Linux)
------------------------------------------------------

- Input: 100 × 60-minute WAV-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)

**Result:** 2702 × real-time (RTF: 0.00037009)

Disk speed:

.. code-block:: txt

   $ sudo hdparm -tT /dev/nvme0n1
     Timing cached reads: 27324 MB in 1.99 seconds = 13707.90 MB/sec
     Timing buffered disk reads: 6880 MB in 3.00 seconds = 2293.17 MB/sec

.. code-block:: txt

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


Run 100 h FLAC-files on NVIDIA Titan RTX (24 GB, Linux)
--------------------------------------------------------

- Input: 100 × 60-minute FLAC-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)

**Result:** 2487 × real-time (RTF: 0.00040214)

.. code-block:: txt

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
