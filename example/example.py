from pathlib import Path

from birdnet import SpeciesPredictions, get_species_from_file, predict_species_within_audio_file
from birdnet.models.v2m4 import AudioModelV2M4Protobuf

audio_path = Path("example/soundscape.wav")
species_path = Path("example/species_list.txt")

# create model instance for v2.4 with language 'en_us'
model = AudioModelV2M4Protobuf(language="en_us")

# predict only the species from this file
custom_species = get_species_from_file(species_path)

# predict species for the whole audio file
predictions = SpeciesPredictions(predict_species_within_audio_file(
  audio_path,
  species_filter=custom_species,
  custom_model=model,
  chunk_overlap_s=2
))

# get predictions at time interval 0s-3s
first_chunk_predictions = list(predictions[(0.0, 3.0)].items())

# get most probable prediction
prediction, confidence = first_chunk_predictions[0]

# get species name
scientific_name, common_name = prediction.split("_")

# print results
print("== Prediction results ==")
print("Chunk start timepoint: 0s")
print("Chunk end timepoint: 3s")
print(f"Scientific name: {scientific_name}")
print(f"Common name: {common_name}")
print(f"Confidence: {confidence*100:.2f}%")
# Output:
# == Prediction results ==
# Chunk start timepoint: 0s
# Chunk end timepoint: 3s
# Scientific name: Poecile atricapillus
# Common name: Black-capped Chickadee
# Confidence: 81.41%
