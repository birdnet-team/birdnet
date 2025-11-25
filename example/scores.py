import birdnet

model = birdnet.load("acoustic", "2.4", "pb")

predictions = model.predict(
  "example/soundscape.wav",
  # predict only the species from the file
  custom_species_list="example/species_list.txt",
)

predictions.to_csv("example/scores.csv")
