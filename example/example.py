from pathlib import Path

import birdnet
from birdnet.utils import get_species_from_file

model = birdnet.load(
  "acoustic", "2.4", "tf", precision="fp32", lang="en_us", library="tf"
)

# predict only the species from the file
predictions = model.predict(
  "example/soundscape.wav",
  custom_species_list=get_species_from_file(Path("example/species_list.txt")),
)

predictions.to_csv("example/predictions.csv")
