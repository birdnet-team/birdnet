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

- Show benchmark options: `birdnet-benchmark --help`
- Predict top 5 species for each segment using CPU und TFLite backend (single file): `birdnet-benchmark soundscape.wav result.csv`
- Predict all audio files in a directory: `birdnet-benchmark path/to/audio/files/ result.csv`
- Use Protobuf backend: `birdnet-benchmark soundscape.wav result.csv -b "pb"`
- Output predictions for top 10 species: `birdnet-benchmark soundscape.wav result.csv --top-k 10 --confidence -100`
- Run on GPU: `birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 1 --device "GPU" --batch-size 1000`
- Run on multiple GPUs: `birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 3 --device "GPU:0" "GPU:1" "GPU:2" --batch-size 1000`
- Increase amount of data feeders: `birdnet-benchmark soundscape.wav /tmp/result.csv --feeders 2`

## Allgemeine Funktionsweise der Python‑Bibliothek

Die Analysepipeline besteht aus fünf logisch getrennten Komponenten:

1. **Ergebnisarray**  
   Eine dreidimensionale Matrix, in der  
   • **Dimension 1** die Eingabedateien,  
   • **Dimension 2** die aufeinanderfolgenden 3‑Sekunden‑Segmente und  
   • **Dimension 3** die im Modell abgebildeten Arten (Spezies)  
   repräsentiert. Jede Matrixzelle enthält die vorhergesagte Wahrscheinlichkeit für die jeweilige Art in dem entsprechenden Segment der Datei.

2. **Buffer**  
   Zwischenspeicher, der Batches von 3‑s‑Audiosegmenten vorhält.

3. **Feeder‑Prozess(e)**  
   Lesen die Eingabedateien, zerlegen sie in 3‑s‑Segmente und füllen den Buffer.

4. **Worker‑Prozess(e)**  
   Entnehmen Batches aus dem Buffer und führen die Inferenz mittels des ML‑Modells aus.

5. **Consumer**  
   Empfängt die von den Workern berechneten Wahrscheinlichkeiten und schreibt sie in das Ergebnisarray.

### Parallelisierung und Ressourcenmanagement

* **Prozessanzahl**  
  Die Anzahl der Feeder‑ und Worker‑Prozesse ist konfigurierbar. Standardmäßig wird ein (1) Feeder gestartet, während die Worker‑Anzahl der Zahl der _physischen_ CPU‑Kerne des Systems entspricht.

* **Buffergröße**  
  Per Default ist der Buffer doppelt so groß wie die Worker‑Anzahl, sodass jeder Worker stets einen vorab geladenen Batch verarbeiten kann und keine Leerlaufzeit entsteht.

* **Modell‑Backends**  
  Jeder Worker lädt das Inferenzmodell separat in seinen Prozess. Auf der CPU können sowohl **TFLite‑** als auch **Protocol‑Buffers‑**Modelle (Protobuf-Modelle) genutzt werden; Protobuf‑Modelle lassen sich optional auch auf der GPU ausführen.

* **Best‑Practice für CPU‑Inferenz**  
  Für reine CPU‑Ausführung sollte die Zahl der Worker‑Prozesse die physische Kernzahl nicht überschreiten, da Oversubscription typischerweise zu einem Leistungsabfall führt.

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

## Interpretation der Laufzeitmetriken

Während der Analyse werden die wichtigsten Leistungskennzahlen einmal pro Sekunde aktualisiert und ausgegeben.

| Kürzel | Bedeutung | Zielwert / Empfehlung |
|--------|-----------|-----------------------|
| **SPEED** | *Beschleunigungsfaktor* relativ zur Echtzeit (real-time, RT). Ein Wert von `2 xRT` bedeutet, dass zehn Minuten Audio in fünf Minuten verarbeitet werden können. Für seine Berechnung werden der Programmstart und das einmalige Modell‑Laden pro Prozess herausgerechnet. Der SPEED‑Wert ergibt sich aus der mittleren Laufzeit aller Worker‑Prozesse im Verhältnis zur Gesamtdauer des bereits verarbeiteten Audios. Zusätzlich wird die Anzahl der 3‑Sekunden‑Segmente pro Sekunde angegeben. | Möglichst hoch; typischerweise ≥ 50 xRT |
| **MEM** | Gesamter Hauptspeicherverbrauch des Python‑Hauptprozesses *plus* aller Subprozesse sowie des genutzten Shared Memory (in MB). | Unterhalb der verfügbaren RAM‑Kapazität halten |
| **BUF** | Durchschnittliche Zahl der Batches im Buffer, dargestellt als *aktuell / maximal*. | Für `W` Worker: `BUF ≈ 2W/2W` |
| **WAIT** | Mittlere Wartezeit (in ms), die Worker auf einen neuen Batch im Buffer warten. | NVMe‑SSDs: ≤ 1 ms |
| **BUSY** | Durchschnittliche Anzahl gleichzeitig ausgelasteter Worker, dargestellt als *aktiv / insgesamt*. | Möglichst `W/W` |
| **PROG** | Gesamtfortschritt der Analyse in %. | Steigt linear von 0 % → 100 % |
| **ETA** | Geschätzte verbleibende Laufzeit bis Abschluss. | möglichst klein |

```text
Beispiel‑Log‑Zeile
------------------
SPEED: 51 xRT [17 seg/s]; MEM: 1590 M; BUF: 8/8; WAIT: 0.17 ms; BUSY: 4/4; PROG: 93.5 %; ETA 0:00:48
```

### Typische Engpässe und Gegenmaßnahmen

* **Hohe WAIT‑Werte oder leerer Buffer**  
  → Anzahl der Feeder erhöhen. Reicht dies nicht, Audiodaten auf ein schnelleres Speichermedium (NVMe/SSD) kopieren oder Worker‑Zahl reduzieren.

* **BUSY kleiner als Worker‑Zahl**  
  → Meist derselbe Engpass wie oben (I/O‑Flaschenhals). Schritte wie oben durchführen.

* **Cache‑Effekt**  
  Da Betriebssysteme gelesene Dateien im RAM zwischenspeichern, steigt SPEED bei einem zweiten Lauf derselben Audiodaten oft deutlich. Zum Benchmarken nur Durchläufe ab dem zweiten Versuch werten.

## Comparative results

### Run 10 h WAV-files on Intel i7-8565U 4-Core with 16 GB RAM (Windows)

- Input: 10x 60 minute WAV-files
- Disk: NVMe SSD (Intel SSDPEKKF010T8L)
- CMD: `birdnet-benchmark test-dataset/test_dataset_100x60min /tmp/result.csv -f 5`

```txt
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
Computational performance:
  50 x real-time (RTF: 0.01992776)
```

### Run 100 h WAV-files on AMD Ryzen 7 3800X 8-Core with 64 GB RAM (Linux)

- Input: 100x 60 minute WAV-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)
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

### Run 100 h WAV-files on NVIDIA Titan RTX with 24 GB (Linux)

- Input: 100x 60 minute WAV-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)
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

### Run 100 h FLAC-files on NVIDIA Titan RTX with 24 GB (Linux)

- Input: 100x 60 minute FLAC-files
- Disk: NVMe SSD (Samsung MZVLB1T0HBLR-00000)
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

## Benchmarking

- Location of log file:
  - Windows: `C:\Users\{user}\AppData\Local\Temp\birdnet.log`
  - Linux/MacOS: `/tmp/birdnet.log`
