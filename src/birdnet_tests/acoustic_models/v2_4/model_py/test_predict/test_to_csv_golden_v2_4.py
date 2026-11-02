"""A model-content golden: the CSV of a pb run over the short test file.

The CSV writer's bytes are pinned by
``inference/predictions/prediction_result_py/test_to_csv_exact_content.py``;
what this pins is the model's output through the whole pipeline at speed 0.5
with overlap. The input path and the confidences are dropped before comparing:
the path differs per machine, the last digits of a score per runtime.
Regenerate the golden by running this file after an intended change.
"""

from pathlib import Path

import numpy as np

from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_SHORT

GOLDEN_PATH = Path(__file__).with_suffix(".csv")


def remove_input_and_confidence(file_path: Path) -> None:
  import pandas as pd

  df = pd.read_csv(file_path, encoding="utf-8")
  df = df.drop(columns=["input", "confidence"])
  df.to_csv(file_path, index=False, encoding="utf-8")


def predict_to_csv(out_path: Path) -> None:
  model = load("acoustic", "2.4", "pb")
  with model.predict_session(
    n_workers=1,
    top_k=2,
    speed=0.5,
    overlap_duration_s=0.5,
    default_confidence_threshold=-np.inf,
  ) as session:
    res = session.run(TEST_FILE_SHORT)
  res.to_csv(out_path, encoding="utf-8")
  remove_input_and_confidence(out_path)


def test_pb_csv_matches_the_golden(tmp_path: Path) -> None:
  out_path = tmp_path / "out.csv"
  predict_to_csv(out_path)
  assert out_path.read_text(encoding="utf-8") == GOLDEN_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
  predict_to_csv(GOLDEN_PATH)
