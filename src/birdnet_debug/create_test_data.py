import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Load the audio file
INPUT_FILE = Path("example/soundscape.wav")
INPUT_FILE_DUR = 2
DURATION_PER_FILE = 60  # min
N_FILES = 100
DTYPE = ".flac"

OUTPUT_FOLDER_BASE = Path("/data/test-dataset")
OUTPUT_FOLDER_BASE = Path("/home/mi/tmp")
OUTPUT_FOLDER_BASE = Path("test-dataset")

REPS = round(DURATION_PER_FILE / INPUT_FILE_DUR)
OUTPUT_FOLDER = OUTPUT_FOLDER_BASE / f"test_dataset_{N_FILES}x{DURATION_PER_FILE}min"
if OUTPUT_FOLDER.is_dir():
  shutil.rmtree(OUTPUT_FOLDER)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Read the audio data and sample rate from the input file
data, samplerate = sf.read(INPUT_FILE)


repeated_data = np.tile(data, REPS)

with NamedTemporaryFile("w", suffix=DTYPE) as f:
  ref = Path(f.name)
  sf.write(ref, repeated_data, samplerate)
  del repeated_data
  for file_nr in tqdm(range(N_FILES)):
    output_file = OUTPUT_FOLDER / f"{file_nr:0{len(str(N_FILES))}d}.wav"
    shutil.copyfile(ref, output_file)
print(OUTPUT_FOLDER)
