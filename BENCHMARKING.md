# Birdnet benchmark command line tool

## Install

Get newest `birdnet` version via TUCcloud: [https://tuc.cloud/index.php/s/Qtace7JWnTKAe88](https://tuc.cloud/index.php/s/Qtace7JWnTKAe88)

A Python 3.11 installation is required. If you don't have it, you can install it from [python.org](https://www.python.org/downloads/release/python-3119/).

### Preparation on Windows (CMD)

```cmd
py -3.11 -m venv .venv-bn
.venv-bn\\Scripts\\activate
python.exe -m pip install --upgrade pip
python.exe -m pip install wheel
python.exe -m pip install birdnet-0.2.0a0-py3-none-any.whl
```

### Preparation on Linux (Bash)

```sh
python3.11 -m venv .venv-bn
source .venv-bn/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install birdnet-0.2.0a0-py3-none-any.whl
```

### Install with GPU support

Use the suffix `[and-cuda]` while installing, i.e., `... pip install birdnet-0.2.0a0-py3-none-any.whl[and-cuda]` to support running the benchmark on a GPU with CUDA support.

### Upgrade version

Just install the new version in the activated environment.

## Example usage

### Show benchmark options

`birdnet-benchmark --help`

### Predict top 5 species for each segment using CPU und TFLite backend (single file)

`birdnet-benchmark soundscape.wav result.csv`

### Predict all audio files in a directory

`birdnet-benchmark path/to/audio/files/ result.csv`

### Use Protobuf backend

`birdnet-benchmark soundscape.wav result.csv -b "pb"`

### Output predictions for top 10 species

`birdnet-benchmark soundscape.wav result.csv --top-k 10 --confidence -100`

### Run on single GPU

`birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 1 --device "GPU" --batch-size 1000`

### Run on multiple GPUs

`birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 3 --device "GPU:0" "GPU:1" "GPU:2" --batch-size 1000`

### Increase amount of data feeders

`birdnet-benchmark soundscape.wav /tmp/result.csv --feeders 2`

## General function of the Python library

Die Analyse funktioniert wie folgt. Sie besteht grob aus fünf Teilen:

1. Ergebnisarray: Eine 3-dimensionale Tabelle, die für jede Eingabedatei (Dimension 1) und für jedes 3-Sekundensegment (Dimension 2) für alle Spezies des Modells (Dimension 3) die entsprechenden Wahrscheinlichkeiten enthält.
1. Buffer: Hält Batches von drei Sekundenlangen Audiosegmenten
2. Feeder(s): Prozesse, die den Buffer mit Batches füllen, indem sie die Eingabedateien lesen und in Segmente aufteilen
3. Worker(s): Prozesse, die die Batches aus dem Buffer holen und die Analyze durchführen
4. Consumer: Holt die Analyseergebnisse der Workers und schreibt sie in das Ergebnissarray

Die Anzahl der Feeders und Workers kann hierbei dynamisch angepasst werden, im Standardfall wird jedoch ein Feeder verwendet und die Anzahl an Workern ist die Anzahl an physischen CPU Kernen des Computers.
Der Buffer hat im Standardfall die doppelte Größe wie die Anzahl an Workern, damit immer ein neuer Batch pro Worker vorgeladen ist, und diese nicht auf neue Segmente warten müssen.
Die Worker können hierbei auf der CPU entweder mittels TFLite oder Protobuf Modell laufen. Jeder Worker lädt hierbei separat das Modell in seinen Prozess. Das Protobufmodell kann alternativ auch auf der GPU laufen.
Diese Zahl der Worker sollte bei der Synthese auf der CPU nicht die Anzahl an physischen Kernen überschritten werden, da es sich in der Regel eher negativ auf die Performanz auswirkt.

## Interpreting the numbers while analysis

Every second there will be an update with the progress of the analysis. It will display following values:

- SPEED: the computation speed expressed as speed times real-time, e.g., a value of 2 will say, the computation can process ten minutes of audio in five minutes. Hierbei wird der Programmstart und die Zeit für das Laden des Modells pro Prozess herausgerechnet. Der Wert ist also höher als wenn die gesamte Laufzeit berechnen würde. Der Wert berechnet sich aus der durchschnittlichen Laufzeit aller Arbeiterprozesse und der Gesamtdauer des verarbeiteten Audios. There will be also the amount of three second segments of the input file(s), that are processed per second.
- MEM: this value will display the total memory consumption of the whole process in MB, including all subprocesses and used shared memory.
- BUF: gibt an, wieviele Batches sich durchschnittlich im Buffer befinden.
- WAIT: gibt an, wie lange die Arbeiter durchschnittlich auf einen neuen Batch warten müssen.
- BUSY: gibt an, wie viele Arbeiter druchschnittlich gleichzeitig aktiv sind und nicht auf einen Batch warten.
- PROG: gibt den gesamten Fortschritt der Analyse an
- ETA: gibt die geschätze Restzeit für die Analyse an

Der Wert für WAIT sollte so niedrig wie möglich sein (NVMe <=0.05 ms) und der Buffer sollte durchschnittlich immer gefüllt sein, also z.B. für 4 Worker `BUF: 8/8` anzeigen. Falls dies nicht der Fall ist, dann sollte die Anzahl an Feedern erhöht werden und falls dies nicht hilft, dann sollten die Daten auf ein schnelleres Speichermedium kopiert und von dort gelesen werden. Man kann alternativ auch die Anzahl an Arbeitern verringern, aber verliert dadurch an Geschwindigkeit, denn je mehr Feeder aktiv sind, desto mehr könnten die Arbeiter auch beeinträchtigt werden.

Die aktiven Arbeiter erhöhen sich anfangs allmälig und sollten immer alle aktiv sein, also z.B. für 4 Worker `BUSY: 4/4` anzeigen. Wenn es nicht der Fall sein sollte, ist die Datengrundlage wieder das Problem, und es sollten die gleichen Schritte wie soeben erwähnt unternommen werden.

Es ist noch zu beachten, dass das die Audiodaten vom Computer gecached werden, und die Analyse daher schneller ist, wenn man sie mehrfach hintereinander laufen lässt, daher sollte der erste Versuch nicht gezählt werden.

## Interpreting the numbers after analysis

After you've run the analysis you got the benchmarking results:

- Wall time: Zeit, die von kompletten Programmstart bis Ende verbraucht wurde.
- Buffer: Zeigt an wieviele Batches durchschnittlich im Buffer waren
- Busy workers: Zeigt an, wieviele Arbeiter durchschnittlich aktiv waren und die durchschnittliche Dauer um auf den nächsten Batch zu warten.
- Memory usage: Zeit den maximalen Ramverbrauch des Programms an, die Größe des Buffers und die Größe des Ergebnisarrays
- Performance: Gibt die Geschwindigkeit an in Faktor mal Echtzeit. Berechnet sich aus der Wall time und der Anzahl an verarbeiteten Stunden. Es wird auch die Anzahl an Segmenten pro Sekunde angezeigt und die Menge an Audio, die pro Sekunde verarbeitet wird.
- Computational performance: Zeit an, wie schnell die Berechnung war. Der Wert stellt den gleichen Wert dar wie SPEED nur nach Ende der Analyse.

## Comparative results

### Run 100 h on NVidia Titan RTX with 24 GB (Linux, WAV-files)

- Input: 100x 60 minute WAV-files
- Disk: NVMe SSD
- CMD: `birdnet-benchmark test-dataset/test_dataset_100x60min /tmp/result.csv --device GPU --backend pb -w 1 -f 5 -s 1025`

```txt
-------------------------------
------ Benchmark summary ------
-------------------------------
Start time: 07/09/2025 02:47 PM
End time:   07/09/2025 02:49 PM
Wall time:  0:02:24.357884
Input: 100 file(s) (WAV)
  Total duration: 4 days, 4:00:00
  Average duration: 1:00:00
  Minimum duration (single file): 1:00:00
  Maximum duration (single file): 1:00:00
Feeder(s): 5
Buffer: 1.9/2 filled slots (mean)
Busy workers: 1.0/1 (mean)
  Average wait time for next batch: 0.051 ms
Memory usage:
  Program: 9368.30 M (total max)
  Buffer: 1126.11 M (shared memory)
  Result: 2.94 M (NumPy)
Performance:
  2494 x real-time (RTF: 0.00040099)
  831 segments/s (0:41:33.802132 audio/s)
Computational performance:
  2614 x real-time (RTF: 0.00038252)
```

### Run 100 h on NVidia Titan RTX with 24 GB (Linux, FLAC-files)

- Input: 100x 60 minute FLAC-files
- Disk: NVMe SSD
- CMD: `birdnet-benchmark test-dataset/test_dataset_100x60min_flac /tmp/result.csv --device GPU --backend pb -w 1 -f 5 -s 1025`

```txt
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
Computational performance:
  2602 x real-time (RTF: 0.00038431)
```

### Run 100 h on AMD Ryzen 7 3800X 8-Core with 64 GB RAM (Linux, WAV-files)

- Input: 100x 60 minute WAV-files
- Disk: NVMe SSD
- CMD: `birdnet-benchmark test-dataset/test_dataset_100x60min /tmp/result.csv -f 5`

```txt
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
Computational performance:
  417 x real-time (RTF: 0.00239808)
```