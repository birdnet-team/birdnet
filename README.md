# birdnet

[![PyPI](https://img.shields.io/pypi/v/birdnet.svg)](https://pypi.python.org/pypi/birdnet)
![PyPI](https://img.shields.io/pypi/pyversions/birdnet.svg)
[![MIT](https://img.shields.io/github/license/birdnet-team/birdnet.svg)](https://github.com/birdnet-team/birdnet/blob/main/LICENSE.md)

A Python library for identifying bird species by their sounds.

The library is geared towards providing a robust workflow for ecological data analysis in bioacoustic projects. While it covers essential functionalities, it doesn’t include all the features found in BirdNET-Analyzer, which is available [here](https://github.com/kahst/BirdNET-Analyzer). Some features might only be available in the BirdNET Analyzer and not in this package.

Please note that the project is under active development, so you might encounter changes that could affect your current workflow. We recommend checking for updates regularly.

The package is also available as an R package at: [birdnetR](https://github.com/birdnet-team/birdnetR).

## Installation

```sh
# For CPU users
pip install birdnet --user

# For GPU users (NVIDIA GPU driver and CUDA need to be installed in advance)
pip install birdnet[and-cuda] --user
```

## Example usage

### Identify species within an audio file

```py
from pathlib import Path

from birdnet import SpeciesPredictions, predict_species_within_audio_file

# predict species within the whole audio file
audio_path = Path("example/soundscape.wav")
predictions = SpeciesPredictions(predict_species_within_audio_file(audio_path))

# get most probable prediction at time interval 0s-3s
prediction, confidence = list(predictions[(0.0, 3.0)].items())[0]
print(f"predicted '{prediction}' with a confidence of {confidence:.2f}")
# output:
# predicted 'Poecile atricapillus_Black-capped Chickadee' with a confidence of 0.81
```

The resulting `predictions` look like this (excerpt, scores may vary):

```py
from birdnet import SpeciesPredictions, SpeciesPrediction

predictions = SpeciesPredictions([
  ((0.0, 3.0), SpeciesPrediction([
    ('Poecile atricapillus_Black-capped Chickadee', 0.8140561)
  ])),
  ((3.0, 6.0), SpeciesPrediction([
    ('Poecile atricapillus_Black-capped Chickadee', 0.3082859)
  ])),
  ((6.0, 9.0), SpeciesPrediction([
    ('Baeolophus bicolor_Tufted Titmouse', 0.1864328)
  ])),
  ((9.0, 12.0), SpeciesPrediction([
    ('Haemorhous mexicanus_House Finch', 0.639378)
  ])),
  ((12.0, 15.0), SpeciesPrediction()),
  ((15.0, 18.0), SpeciesPrediction()),
  ((18.0, 21.0), SpeciesPrediction([
    ('Cyanocitta cristata_Blue Jay', 0.4352715),
    ('Clamator coromandus_Chestnut-winged Cuckoo', 0.32258758)
  ])),
  ((21.0, 24.0), SpeciesPrediction([
    ('Cyanocitta cristata_Blue Jay', 0.32908556),
    ('Haemorhous mexicanus_House Finch', 0.18672176)
  ])),
  ...
])
```

For a more detailed prediction you can take a look at [example/example.py](./example/example.py).

### Predict species for a given location and time

```py
from birdnet import predict_species_at_location_and_time

# predict species
prediction = predict_species_at_location_and_time(42.5, -76.45, week=4)

# get most probable species
first_species, confidence = list(prediction.items())[0]
print(f"predicted '{first_species}' with a confidence of {confidence:.2f}")
# output:
# predicted 'Cyanocitta cristata_Blue Jay' with a confidence of 0.93
```

### Identify species within audio files using multiprocessing

```py
from pathlib import Path

from birdnet import predict_species_within_audio_files_mp

files = (
  Path("example/soundscape.wav"),
  Path("example/soundscape.wav"),
  Path("example/soundscape.wav"),
  Path("example/soundscape.wav"),
)

file_predictions = list(predict_species_within_audio_files_mp(files))

for file, predictions in file_predictions:
  print(file.name, len(predictions), "predictions")
# output:
# soundscape.wav 40 predictions
# soundscape.wav 40 predictions
# soundscape.wav 40 predictions
# soundscape.wav 40 predictions
```

### File formats

The audio models support all formats compatible with the SoundFile library (see [here](https://python-soundfile.readthedocs.io/en/0.11.0/#read-write-functions)). This includes, but is not limited to, WAV, FLAC, OGG, and AIFF. The flexibility of supported formats ensures that the models can handle a wide variety of audio input types, making them adaptable to different use cases and environments.

### Model Formats and Execution Details

This project provides two model formats: Protobuf/Raven and TFLite. Both models are designed to have identical precision up to 2 decimal places, with differences only appearing from the third decimal place onward.

- **Protobuf Model**: Accessed via `AudioModelV2M4Protobuf()`/`MetaModelV2M4Protobuf()`/`CustomAudioModelV2M4Raven()`, this model can be executed on both GPU and CPU. By default, the Protobuf model is used, and the system will attempt to run it on the GPU if available.
- **TFLite Model**: Accessed via `AudioModelV2M4TFLite()`/`MetaModelV2M4TFLite()`/`CustomAudioModelV2M4TFLite()`, this model is limited to CPU execution only.

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

The German Federal Ministry of Education and Research is funding the development of BirdNET through the project "BirdNET+" (FKZ 01|S22072).
Additionally, the German Federal Ministry of Environment, Nature Conservation and Nuclear Safety is funding the development of BirdNET through the project "DeepBirdDetect" (FKZ 67KI31040E).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
