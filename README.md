# birdnet

[![PyPI](https://img.shields.io/pypi/v/birdnet.svg)](https://pypi.python.org/pypi/birdnet)
![PyPI](https://img.shields.io/pypi/pyversions/birdnet.svg)
[![MIT](https://img.shields.io/github/license/birdnet-team/birdnet.svg)](https://github.com/birdnet-team/birdnet/blob/main/LICENSE.md)

A Python library for identifying bird species by their sounds.

## Installation

```sh
pip install birdnet --user
```

## Example usage

### Identify species within an audio file

```py
from pathlib import Path

from birdnet.models import ModelV2M4

# create model instance for v2.4
model = ModelV2M4()

# predict species within the whole audio file
audio_path = Path("example/soundscape.wav")
predictions = model.predict_species_within_audio_file(audio_path)

# get most probable prediction at time interval 0s-3s
prediction, confidence = list(predictions[(0.0, 3.0)].items())[0]
print(f"predicted '{prediction}' with a confidence of {confidence}")
# Output:
# predicted 'Poecile atricapillus_Black-capped Chickadee' with a confidence of 0.8140556812286377
```

For a more detailled prediction you can take a look at [example/minimal_example.py](./example/minimal_example.py).

### Predict species for a given location and time

```py
from birdnet.models import ModelV2M4

# create model instance for v2.4
model = ModelV2M4()

# predict species
predictions = model.predict_species_at_location_and_time(42.5, -76.45, week=4)

# get most probable prediction
first_prediction, confidence = list(predictions.items())[0]
print(f"predicted '{first_prediction}' with a confidence of {confidence}")
# Output:
# predicted 'Cyanocitta cristata_Blue Jay' with a confidence of 0.9276198744773865
```

## Citation

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

This project is supported by Jake Holshuh (Cornell class of `'69) and The Arthur Vining Davis Foundations. Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The German Federal Ministry of Education and Research is funding the development of BirdNET through the project "BirdNET+" (FKZ 01|S22072).
Additionally, the German Federal Ministry of Environment, Nature Conservation and Nuclear Safety is funding the development of BirdNET through the project "DeepBirdDetect" (FKZ 67KI31040E).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
