import tempfile
from pathlib import Path

from birdnet.acoustic_models.inference.prediction_result import (
  PredictionResult,
)

if __name__ == "__main__":
  res = PredictionResult.load(
    Path(tempfile.gettempdir()) / "soundscape.npz"
  )  # Example usage

  df = res.to_csv(
    Path(tempfile.gettempdir()) / "soundscape-out.csv",
    encoding="utf-8",
  )

  # print(Path(tempfile.gettempdir()) / "predictions.csv")

  # res = load_prediction_data(Path(tempfile.gettempdir()) / "predictions.npz")
  # print(res)
