import birdnet

model = birdnet.load("geo", "2.4", "tf", lang="en_us", library="tf")

predictions = model.predict(42.5, -76.45, week=4)

predictions.to_csv("example/location.csv")
