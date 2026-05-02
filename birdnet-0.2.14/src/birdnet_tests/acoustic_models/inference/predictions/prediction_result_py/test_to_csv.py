from pathlib import Path

import numpy as np

from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG

OUT_PATH = Path(__file__).with_suffix(".csv")


def remove_input_and_confidence(file_path: Path) -> None:
  import pandas as pd

  df = pd.read_csv(file_path, encoding="utf-8")
  df = df.drop(columns=["input", "confidence"])
  df.to_csv(file_path, index=False, encoding="utf-8")


def create_output() -> None:
  model = load("acoustic", "2.4", "pb")
  with model.predict_session(
    top_k=2, speed=0.5, overlap_duration_s=0.5, default_confidence_threshold=-np.inf
  ) as session:
    res = session.run(TEST_FILE_LONG)
  res.to_csv(OUT_PATH, encoding="utf-8")
  remove_input_and_confidence(OUT_PATH)


def test_full_pipeline(tmp_path: Path) -> None:
  model = load("acoustic", "2.4", "pb")
  with model.predict_session(
    top_k=2, speed=0.5, overlap_duration_s=0.5, default_confidence_threshold=-np.inf
  ) as session:
    res = session.run(TEST_FILE_LONG)
  tmp_file_path = tmp_path.with_suffix(".csv")
  res.to_csv(tmp_file_path, encoding="utf-8")
  remove_input_and_confidence(tmp_file_path)
  test_content = tmp_file_path.read_text(encoding="utf-8")
  reference_content = OUT_PATH.read_text(encoding="utf-8")
  assert reference_content == test_content


if __name__ == "__main__":
  create_output()
