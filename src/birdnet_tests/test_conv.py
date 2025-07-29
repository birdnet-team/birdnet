import tempfile
from hashlib import sha1
from pathlib import Path

from birdnet.acoustic_models.inference.scores.prediction_result import (
  PredictionResult,
)
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_WAV


def get_cached_result(audio_paths: list[Path], k: int, conf: float) -> PredictionResult:
  tmp_dir = Path(tempfile.gettempdir()) / "birdnet_tests" / "test_conv"
  tmp_dir.mkdir(parents=True, exist_ok=True)
  # get short name from list audio_paths
  name = f"t{k}_c{conf}_"
  name += "_".join(str(p.absolute()) for p in audio_paths)
  name_short = sha1(name.encode("utf-8")).hexdigest()[:20]
  npz_path = tmp_dir / f"{name_short}.npz"
  if npz_path.is_file():
    return PredictionResult.load(npz_path)
  else:
    model = load("acoustic", "2.4", "tf")
    result = model.predict(
      audio_paths,
      top_k=k,
      default_confidence_threshold=conf,
      workers=12,
    )
    result.save(npz_path)
    return result


def test_soundscape() -> None:
  audio_path = [TEST_FILE_WAV]

  result = get_cached_result(audio_path, 5, 0.1)
  array = result.to_structured_array()
  assert array.shape == (36,)
  assert array.dtype.names == (
    "file_path",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  )
