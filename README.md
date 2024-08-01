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
print(f"predicted '{prediction}' with a confidence of {confidence:.6f}")
# output:
# predicted 'Poecile atricapillus_Black-capped Chickadee' with a confidence of 0.814056
```

The resulting `predictions` look like this (excerpt):

```py
predictions = OrderedDict([
  ((0.0, 3.0), OrderedDict([
    ('Poecile atricapillus_Black-capped Chickadee', 0.8140561)
  ])),
  ((3.0, 6.0), OrderedDict([
    ('Poecile atricapillus_Black-capped Chickadee', 0.3082859)
  ])),
  ((6.0, 9.0), OrderedDict([
    ('Baeolophus bicolor_Tufted Titmouse', 0.1864328)
  ])),
  ((9.0, 12.0), OrderedDict([
    ('Haemorhous mexicanus_House Finch', 0.639378)
  ])),
  ((12.0, 15.0), OrderedDict()),
  ((15.0, 18.0), OrderedDict()),
  ((18.0, 21.0), OrderedDict([
    ('Cyanocitta cristata_Blue Jay', 0.4352715),
    ('Clamator coromandus_Chestnut-winged Cuckoo', 0.32258758)
  ])),
  ((21.0, 24.0), OrderedDict([
    ('Cyanocitta cristata_Blue Jay', 0.32908556),
    ('Haemorhous mexicanus_House Finch', 0.18672176)
  ])),
  ...
])
```

For a more detailed prediction you can take a look at [example/example.py](./example/example.py).

### Predict species for a given location and time

```py
from birdnet.models import ModelV2M4

# create model instance for v2.4
model = ModelV2M4()

# predict species
predictions = model.predict_species_at_location_and_time(42.5, -76.45, week=4)

# get most probable prediction
first_prediction, confidence = list(predictions.items())[0]
print(f"predicted '{first_prediction}' with a confidence of {confidence:.6f}")
# output:
# predicted 'Cyanocitta cristata_Blue Jay' with a confidence of 0.927620
```

### Predict species within an audio file for a given location and time

```py
from pathlib import Path

from birdnet.models import ModelV2M4

# create model instance for v2.4
model = ModelV2M4()

# predict species within the whole audio file
audio_path = Path("example/soundscape.wav")

species_in_area = model.predict_species_at_location_and_time(42.5, -76.45, week=4)
predictions = model.predict_species_within_audio_file(
  audio_path,
  filter_species=set(species_in_area.keys())
)

# get most probable prediction at time interval 0s-3s
prediction, confidence = list(predictions[(0.0, 3.0)].items())[0]
print(f"predicted '{prediction}' with a confidence of {confidence:.6f}")
# output:
# predicted 'Poecile atricapillus_Black-capped Chickadee' with a confidence of 0.814056
```

## License

- **Source Code**: The source code for this project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
- **Models**: The models used in this project are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Please ensure you review and adhere to the specific license terms provided with each model. Note that educational and research purposes are considered non-commercial use cases.

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

This project is supported by Jake Holshuh (Cornell class of '69) and The Arthur Vining Davis Foundations. Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The German Federal Ministry of Education and Research is funding the development of BirdNET through the project "BirdNET+" (FKZ 01|S22072).
Additionally, the German Federal Ministry of Environment, Nature Conservation and Nuclear Safety is funding the development of BirdNET through the project "DeepBirdDetect" (FKZ 67KI31040E).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
