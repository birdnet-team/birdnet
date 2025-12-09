# birdnet

<!-- [![CI](https://github.com/birdnet-team/birdnet/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/birdnet-team/birdnet/actions/workflows/ci.yml)  -->
[![PyPI](https://img.shields.io/pypi/v/birdnet.svg)](https://pypi.python.org/pypi/birdnet)
![PyPI](https://img.shields.io/pypi/pyversions/birdnet.svg)
[![MIT](https://img.shields.io/github/license/birdnet-team/birdnet.svg)](https://github.com/birdnet-team/birdnet/blob/main/LICENSE.md)

A Python library for identifying bird species by their sounds.

The library is geared towards providing a robust workflow for ecological data analysis in bioacoustic projects. While it covers essential functionalities, it doesn’t include all the features found in BirdNET-Analyzer, which is available [here](https://github.com/birdnet-team/BirdNET-Analyzer). Some features might only be available in the BirdNET Analyzer and not in this package.

> [!NOTE]
> This project is under active development, so you might encounter changes that could affect your current workflow. We recommend checking for updates regularly.

The package is also available as an R package at: [birdnetR](https://github.com/birdnet-team/birdnetR).

## Features

* 🐦 Extract **classification scores and embeddings** for 6,522 species from audio recordings
* 📍 Predict 6,522 **species presence** for a given location and time
* 🧠 Utilize your **custom-trained acoustic models** from BirdNET-Analyzer
* ⚙️ Support for both **CPU and GPU** execution (including multiple GPUs at the same time)
* 🚀 **Multiprocessing** support for fast batch analysis of large datasets
* 💾 **Low memory footprint** and small disk space requirements
* 🎵 Support for **various audio file formats** (WAV, FLAC, OGG, MP3, etc.)
* 📊 Export results in **various output file formats** (CSV, Arrow table, Parquet, Numpy, etc.)
* 💻 **Cross-platform**: Windows, macOS, and Linux
* 🌍 Use **multilingual** species names (English, German, French, Spanish, etc.)
* ⬇️ **Auto-download** of all official models
* 🛜 Full **offline usage** using local (custom) model files

*The library is optimized for a minimal memory footprint and maximum scalability, making it suitable for both edge devices and high-performance computing clusters.*

## Speed benchmarks

| Device             | Specs       | Disk | OS      | Recordings/s | → 1 h of recording |
|--------------------|-------------|------|---------|--------------|---|
| Intel i7 8th Gen   | 4 cores     | NVMe | Windows | 50 s         | 72 s
| Ryzen 7 3800X      | 8 cores     | NVMe | Linux   | 7 min        | 8.5 s
| Nvidia Titan RTX   | 24 GB VRAM  | NVMe | Linux   | 41 min       | 1.5 s

For more detailed benchmarks, please refer to the [BENCHMARKING.md](BENCHMARKING.md) file.

## Installation

### Platform support and Python versions

| Platform | Architecture | ProtoBuf-CPU | ProtoBuf-GPU | TFLite | LiteRT |
| ----------- | ------------ | ---------------- | ---------------- | ---------------- | ---------------- |
| **Linux** | x86_64 | 3.11, 3.12, 3.13 | 3.11, 3.12, 3.13 | 3.11, 3.12, 3.13 | 3.11, 3.12 |
| | ARM64 | 3.11, 3.12, 3.13 | / | 3.11, 3.12, 3.13 | 3.11, 3.12 |
| **MacOS** | x86_64 | 3.11, 3.12 | / | 3.11, 3.12 | / |
| | ARM64 | 3.11, 3.12, 3.13 | / | 3.11, 3.12, 3.13 | / |
| **Windows** | x86_64 | 3.11, 3.12, 3.13 | / | 3.11, 3.12, 3.13 | / |
| | ARM64 | / | / | / | / |

For details see the official [TensorFlow](https://www.tensorflow.org/install/pip#package_location) documentation.

### Instructions

```sh
# For CPU users
pip install birdnet --user

# For GPU users (NVIDIA GPU driver and CUDA need to be installed in advance)
pip install birdnet[and-cuda] --user
```

If you encounter issues with audio file reading, please ensure that `libsndfile` is installed on your system.

- **Ubuntu/Debian**: `sudo apt-get install libsndfile1`
- **macOS** (using Homebrew): `brew install libsndfile`
- **Windows**: Download and install the precompiled binaries from the [official website](https://github.com/libsndfile/libsndfile/releases/), extract them and add the folder to path.

## Supported operations, precisions and devices

### V2.4

| **Model** | Acoustic | Acoustic | Acoustic | Geo | Geo |
|---|---|---|---|---|---|
| **Backend** | TFLite/<br>LiteRT | ProtoBuf | ProtoBuf<br>Raven* | TFLite/<br>LiteRT | ProtoBuf |
| `predict(..)` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `encode(..)` | ✅ | ✅ | ❌ | ❌ | ❌ |
| **INT8** | CPU | ❌ | ❌ | ❌ | ❌ |
| **FP16** | CPU | ❌ | ❌ | ❌ | ❌ |
| **FP32** | CPU | CPU/GPU | CPU/GPU | CPU | CPU/GPU |

✅ = Supported ❌ = Not supported\
**ProtoBuf Raven is only available for custom acoustic models.*

### Perch V2

| **Model** | Acoustic | Geo
|---|---|---|
| **Backend** | ProtoBuf | ❌
| `predict(..)` | ✅ | ❌
| `encode(..)` | ✅ | ❌
| **INT8** | ❌ | ❌
| **FP16** | ❌ | ❌
| **FP32** | CPU/GPU | ❌

## Example usage

### Identify species within an audio file

```py
import birdnet

model = birdnet.load("acoustic", "2.4", "tf")

predictions = model.predict(
  "example/soundscape.wav",
  # predict only the species from the file
  custom_species_list="example/species_list.txt",
)

predictions.to_csv("example/predictions.csv")
```

The resulting predictions look like this (excerpt, scores may vary):

|input|start_time|end_time|species_name|confidence|
|---|---|---|---|---|
|/home/.../example/soundscape.wav|00:00:00.00|00:00:03.00|Poecile atricapillus_Black-capped Chickadee|0.814|
|/home/.../example/soundscape.wav|00:00:03.00|00:00:06.00|Poecile atricapillus_Black-capped Chickadee|0.3084|
|/home/.../example/soundscape.wav|00:00:06.00|00:00:09.00|Baeolophus bicolor_Tufted Titmouse|0.1864|
|/home/.../example/soundscape.wav|00:00:09.00|00:00:12.00|Haemorhous mexicanus_House Finch|0.6392|
|/home/.../example/soundscape.wav|00:00:18.00|00:00:21.00|Cyanocitta cristata_Blue Jay|0.4353|
|/home/.../example/soundscape.wav|00:00:21.00|00:00:24.00|Cyanocitta cristata_Blue Jay|0.3291|
|/home/.../example/soundscape.wav|00:00:21.00|00:00:24.00|Haemorhous mexicanus_House Finch|0.1866|
|...|...|...|...|...|

For a more detailed prediction you can take a look at [example/predictions.csv](example/predictions.csv).

### Predict species for a given location and time

```py
import birdnet

model = birdnet.load("geo", "2.4", "tf")

predictions = model.predict(42.5, -76.45, week=4)

predictions.to_csv("example/location.csv")
```

The resulting predictions look like this (excerpt, scores may vary; sorted alphabetically):

| species_name                          | confidence |
|---------------------------------------|------------|
| Acanthis flammea_Common Redpoll       | 0.0442     |
| Accipiter cooperii_Cooper's Hawk      | 0.0812     |
| Agelaius phoeniceus_Red-winged Blackbird | 0.0996  |
| Anas platyrhynchos_Mallard            | 0.4468     |
| Anas rubripes_American Black Duck     | 0.11       |
| ... | ... |

The full result is at [example/location.csv](example/location.csv).

### Location of Log File

If something goes wrong, you can find the log file in the following locations:

- Windows: `C:\Users\{user}\AppData\Local\Temp\birdnet.log`
- Linux/MacOS: `/tmp/birdnet.log`

## File formats

The audio models support all formats compatible with the SoundFile library (see [here](https://python-soundfile.readthedocs.io/en/0.11.0/#read-write-functions)). This includes, but is not limited to, WAV, FLAC, OGG, and AIFF. The flexibility of supported formats ensures that the models can handle a wide variety of audio input types, making them adaptable to different use cases and environments.

- Supported: AIFC, AIFF, AU, AVR, CAF, FLAC, HTK, IRCAM, MAT4, MAT5, MP3, MPC2K, NIST, OGG, OPUS, PAF, PVF, RAW, RF64, SD2, SDS, SVX, VOC, W64, WAV, WAVEX, WVE, XI
- Not supportet at the moment: AAC, M4A, WMA

## Model formats and execution details

This project provides two model formats: Protobuf/Raven and TFLite. Both models are designed to have identical precision up to 2 decimal places, with differences only appearing from the third decimal place onward.

- **TFLite Model**: This model is limited to CPU execution only.
- **ProtoBuf Model**: This model can be executed on both GPU and CPU.

Ensure your environment is configured to utilize the appropriate model and available hardware optimally.

## License

- **Source Code**: The source code for this project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
- **Models**: The models used in this project are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Please ensure you review and adhere to the specific license terms provided with each model. Note that educational and research purposes are considered non-commercial use cases.

## Citation

Feel free to use birdnet for your acoustic analyses and research. If you do, please cite as:

```bibtex
@article{kahl2021birdnet,
  title={BirdNET: A deep learning solution for avian diversity monitoring},
  author={Kahl, Stefan and Wood, Connor M and Eibl, Maximilian and Klinck, Holger},
  journal={Ecological Informatics},
  volume={61},
  pages={101236},
  year={2021},
  publisher={Elsevier}
}
```

## Funding

Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Research, Technology and Space (FKZ 01|S22072), the German Federal Ministry for the Environment, Climate Action, Nature Conservation and Nuclear Safety (FKZ 67KI31040E), the German Federal Ministry of Economic Affairs and Energy (FKZ 16KN095550), the Deutsche Bundesstiftung Umwelt (project 39263/01) and the European Social Fund.

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
