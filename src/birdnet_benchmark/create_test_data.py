import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def create_test_dataset(
  input_file: Path,
  output_dir: Path,
  *,
  n_files: int = 10,
  minutes_per_file: int = 10,
  sr: int = 48_000,
  dtype: str = ".wav",
  overwrite: bool = False,
) -> tuple[Path, float]:
  target_dir = (
    output_dir / f"{n_files}x{minutes_per_file}min_{sr / 1_000:.0f}kHz_{dtype[1:]}"
  )
  if target_dir.is_dir():
    if overwrite:
      shutil.rmtree(target_dir)
    else:
      size_mb = sum(f.stat().st_size for f in target_dir.iterdir()) / (1024**2)
      return target_dir, size_mb
  target_dir.mkdir(parents=True, exist_ok=True)

  data, samplerate = sf.read(input_file)

  if samplerate != sr:
    from birdnet.acoustic.inference.core.producer import resample_array_by_sr

    data = resample_array_by_sr(
      data,
      sample_rate=samplerate,
      target_sample_rate=sr,
    )

  reps = round(minutes_per_file / (len(data) / samplerate / 60))
  repeated_data = np.tile(data, reps)

  total_size_mb = 0

  reference_tmp_file = target_dir / "temp_file.wav"
  sf.write(reference_tmp_file, repeated_data, sr)
  ref_file_size_mb = reference_tmp_file.stat().st_size / (1024**2)
  total_size_mb = ref_file_size_mb * n_files
  del repeated_data
  for file_nr in tqdm(range(n_files)):
    target_file = target_dir / f"{file_nr:0{len(str(n_files)) - 1}d}{dtype}"
    shutil.copyfile(reference_tmp_file, target_file)
  reference_tmp_file.unlink()

  return target_dir, total_size_mb
