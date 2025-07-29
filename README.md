# birdnet

[![PyPI](https://img.shields.io/pypi/v/birdnet.svg)](https://pypi.python.org/pypi/birdnet)
![PyPI](https://img.shields.io/pypi/pyversions/birdnet.svg)
[![MIT](https://img.shields.io/github/license/birdnet-team/birdnet.svg)](https://github.com/birdnet-team/birdnet/blob/main/LICENSE.md)

A Python library for identifying bird species by their sounds.

The library is geared towards providing a robust workflow for ecological data analysis in bioacoustic projects. While it covers essential functionalities, it doesn’t include all the features found in BirdNET-Analyzer, which is available [here](https://github.com/birdnet-team/BirdNET-Analyzer). Some features might only be available in the BirdNET Analyzer and not in this package.

Please note that the project is under active development, so you might encounter changes that could affect your current workflow. We recommend checking for updates regularly.

The package is also available as an R package at: [birdnetR](https://github.com/birdnet-team/birdnetR).

## Installation

```sh
# For CPU users
pip install birdnet --user

# For GPU users (NVIDIA GPU driver and CUDA need to be installed in advance)
pip install birdnet[and-cuda] --user

# For edge devices (e.g., Raspberry Pi)
pip install birdnet[litert] --user
```

## Example usage

### Identify species within an audio file

```py
from pathlib import Path

import birdnet
from birdnet.utils import get_species_from_file

model = birdnet.load("acoustic", "2.4", "tf", lang="en_us")

# predict only the species from the file
predictions = model.predict(
  "example/soundscape.wav",
  custom_species_list=get_species_from_file(Path("example/species_list.txt")),
)

predictions.to_csv("example/predictions.csv")
```

The resulting predictions look like this (excerpt, scores may vary):

<img src="example/scores_preview.png" alt="Preview" style="max-width: 700px; height: auto;">

For a more detailed prediction you can take a look at [example/predictions.csv](example/scores.csv).

### Predict species for a given location and time

```py
import birdnet

model = birdnet.load("geo", "2.4", "tf", lang="en_us")

predictions = model.predict(42.5, -76.45, week=4)

predictions.to_csv("example/location.csv")
```

<img src="example/location_preview.png" alt="Preview" style="max-width: 300px; height: auto;">

The result is at [example/location.csv](example/location.csv).

### Location of Log File

If something goes wrong, you can find the log file in the following locations:

- Windows: `C:\Users\{user}\AppData\Local\Temp\birdnet.log`
- Linux/MacOS: `/tmp/birdnet.log`

## Benchmark

For a preliminary benchmark, see [benchmark/BENCHMARK.md](https://github.com/birdnet-team/birdnet/blob/main/benchmark/BENCHMARK.md)

## File formats

The audio models support all formats compatible with the SoundFile library (see [here](https://python-soundfile.readthedocs.io/en/0.11.0/#read-write-functions)). This includes, but is not limited to, WAV, FLAC, OGG, and AIFF. The flexibility of supported formats ensures that the models can handle a wide variety of audio input types, making them adaptable to different use cases and environments.

- Supported: AIFC, AIFF, AU, AVR, CAF, FLAC, HTK, IRCAM, MAT4, MAT5, MP3, MPC2K, NIST, OGG, OPUS, PAF, PVF, RAW, RF64, SD2, SDS, SVX, VOC, W64, WAV, WAVEX, WVE, XI
- Not supportet at the moment: AAC, M4A, WMA 


## Model formats and execution details

This project provides two model formats: Protobuf/Raven and TFLite. Both models are designed to have identical precision up to 2 decimal places, with differences only appearing from the third decimal place onward.

- **TFLite Model**: This model is limited to CPU execution only.
- **Protobuf Model**: This model can be executed on both GPU and CPU.

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

This project is supported by Jake Holshuh (Cornell class of '69) and The Arthur Vining Davis Foundations. Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Education and Research through the project “BirdNET+” (FKZ 01|S22072). The German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection contributes through the “DeepBirdDetect” project (FKZ 67KI31040E). In addition, the Deutsche Bundesstiftung Umwelt supports BirdNET through the project “RangerSound” (project 39263/01).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
