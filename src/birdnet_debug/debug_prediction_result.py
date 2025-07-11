from pathlib import Path

from birdnet.acoustic_models.inference.prediction_result import (
  PredictionResult,
)

if __name__ == "__main__":
  path = Path(
    "/home/stefan/.local/share/birdnet/acoustic-benchmarks/v2.4/lib-v0.2.0a0/run-20250711T155208/result-20250711T155208.npz"
  )
  res = PredictionResult.load(path)  # Example usage

  out = path.with_suffix(".csv")
  df = res.to_csv(
    out,
    encoding="utf-8",
  )
  print(out.absolute())

  # print(Path(tempfile.gettempdir()) / "predictions.csv")

  # res = load_prediction_data(Path(tempfile.gettempdir()) / "predictions.npz")
  # print(res)
