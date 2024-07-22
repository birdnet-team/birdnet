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
