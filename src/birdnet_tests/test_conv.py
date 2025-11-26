
from birdnet.model_loader import load
from birdnet_tests.test_files import TEST_FILE_LONG


# def get_cached_result(
#   audio_paths: list[Path], k: int, conf: float
# ) -> PredictionResultBase:
#   tmp_dir = Path(tempfile.gettempdir()) / "birdnet_tests" / "test_conv"
#   tmp_dir.mkdir(parents=True, exist_ok=True)
#   # get short name from list audio_paths
#   name = f"t{k}_c{conf}_"
#   name += "_".join(str(p.absolute()) for p in audio_paths)
#   name_short = sha1(name.encode("utf-8")).hexdigest()[:20]
#   npz_path = tmp_dir / f"{name_short}.npz"
#   if npz_path.is_file():
#     return PredictionResultBase.load(npz_path)
#   else:
#     result.save(npz_path)
#     return result


def test_soundscape() -> None:
  audio_path = [TEST_FILE_LONG]

  model = load("acoustic", "2.4", "tf")
  result = model.predict(
    audio_path,
    top_k=5,
    default_confidence_threshold=0.1,
  )
  array = result.to_structured_array()
  assert array.shape == (36,)
  assert array.dtype.names == (
    "input",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  )
