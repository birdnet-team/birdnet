import tempfile
from hashlib import sha1
from pathlib import Path

from birdnet.acoustic_models.inference.prediction_result import PredictionResult
from birdnet.model_loader import load
from birdnet_tests.helper import duration_counter, memory_monitor


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


def comp_test_soundscape():
  audio_path = [Path("example/soundscape.wav")]

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


def comp_test_flac():
  audio_path = [
    Path("example/soundscape.wav"),
    Path("test-dataset/test_dataset_1x7.3s_flac/0.flac"),
    Path("test-dataset/test_dataset_100x1.3s_flac/000.flac"),
    Path("test-dataset/test_dataset_1000x0.2s_flac/0000.flac"),
  ]

  result = get_cached_result(audio_path, k=6500, conf=-1)
  array = result.to_structured_array()
  print(array)
  assert array.dtype.names == (
    "file_path",
    "start_time",
    "end_time",
    "species_name",
    "confidence",
  )


def comp_test_flac_pa():
  audio_path = [
    Path("example/soundscape.wav"),
    Path("test-dataset/test_dataset_1x7.3s_flac/0.flac"),
    Path("test-dataset/test_dataset_100x1.3s_flac/000.flac"),
    Path("test-dataset/test_dataset_1000x0.2s_flac/0000.flac"),
  ]

  result = get_cached_result(audio_path, k=6500, conf=-1)
  array = result.to_arrow_table()
  array.to_pandas().to_csv("/tmp/test_conv.csv", index=False)
  print(array)


def comp_test_flac_csv():
  audio_path = [
    Path("example/soundscape.wav"),
    Path("test-dataset/test_dataset_1x7.3s_flac/0.flac"),
    Path("test-dataset/test_dataset_100x1.3s_flac/000.flac"),
    Path("test-dataset/test_dataset_1000x0.2s_flac/0000.flac"),
  ]

  result = get_cached_result(audio_path, k=6500, conf=-1)
  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_csv("/tmp/test_conv.csv")
  print(array)


def comp_test_flac_pd():
  audio_path = [
    Path("example/soundscape.wav"),
    Path("test-dataset/test_dataset_1x7.3s_flac/0.flac"),
    Path("test-dataset/test_dataset_100x1.3s_flac/000.flac"),
    Path("test-dataset/test_dataset_1000x0.2s_flac/0000.flac"),
  ]

  result = get_cached_result(audio_path, k=6500, conf=-1)
  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_dataframe()
  print(array)
  print(duration(), "s")
  print(memory_footprint(), "MB")


def comp_test_flac_parquet():
  audio_path = [
    Path("example/soundscape.wav"),
    Path("test-dataset/test_dataset_1x7.3s_flac/0.flac"),
    Path("test-dataset/test_dataset_100x1.3s_flac/000.flac"),
    Path("test-dataset/test_dataset_1000x0.2s_flac/0000.flac"),
  ]

  result = get_cached_result(audio_path, k=6500, conf=-1)
  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_parquet("/tmp/test_conv.parquet")
  print(array)
  print(duration(), "s")
  print(memory_footprint(), "MB")


def test_large_file():
  audio_path = [Path("test-dataset/test_dataset_4x60min")]

  with duration_counter() as duration, memory_monitor() as memory_footprint:
    result = get_cached_result(audio_path, k=6500, conf=-1)
  print(f"Loading -> duration: {duration()} s; memory: {memory_footprint()} MB")

  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_parquet("/tmp/test_conv.parquet")

  print(f"Parquet -> duration: {duration()} s; memory: {memory_footprint()} MB")

  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_structured_array()

  print(f"Numpy -> duration: {duration()} s; memory: {memory_footprint()} MB")

  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_arrow_table()

  print(f"Arrow -> duration: {duration()} s; memory: {memory_footprint()} MB")

  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_dataframe()

  print(f"DataFrame -> duration: {duration()} s; memory: {memory_footprint()} MB")

  with duration_counter() as duration, memory_monitor() as memory_footprint:
    array = result.to_csv("/tmp/test_conv.csv")

  print(f"CSV -> duration: {duration()} s; memory: {memory_footprint()} MB")


test_large_file()
