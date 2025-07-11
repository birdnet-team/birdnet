import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Load the audio file
INPUT_FILE = Path("example/soundscape.wav")
INPUT_FILE_DUR_S = 0.2
N_FILES = 100000
DTYPE = ".flac"

OUTPUT_FOLDER_BASE = Path("/data/test-dataset")
OUTPUT_FOLDER_BASE = Path("/home/mi/tmp")
OUTPUT_FOLDER_BASE = Path("test-dataset")

OUTPUT_FOLDER = (
  OUTPUT_FOLDER_BASE / f"test_dataset_{N_FILES}x{INPUT_FILE_DUR_S}s_{DTYPE[1:]}"
)
if OUTPUT_FOLDER.is_dir():
  shutil.rmtree(OUTPUT_FOLDER)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Read the audio data and sample rate from the input file
data, samplerate = sf.read(INPUT_FILE)

samples_duration = round(INPUT_FILE_DUR_S * samplerate)
data = data[:samples_duration]

ref = OUTPUT_FOLDER / "tmp.wav"

with NamedTemporaryFile("w", suffix=DTYPE) as f:
  # ref = Path(f.name)
  sf.write(ref, data, samplerate)
  del data
  for file_nr in tqdm(range(N_FILES)):
    output_file = OUTPUT_FOLDER / f"{file_nr:0{len(str(N_FILES))}d}{DTYPE}"
    shutil.copyfile(ref, output_file)
ref.unlink()
print(OUTPUT_FOLDER)
