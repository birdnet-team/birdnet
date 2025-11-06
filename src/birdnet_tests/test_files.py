from pathlib import Path

import numpy

TEST_RESULTS_DIR = Path("src/birdnet_tests/test_results")
TEST_FILES_DIR = Path("src/birdnet_tests/TEST_FILES")
AUDIO_FORMATS_DIR = TEST_FILES_DIR / "audio_formats"

# Duration: 120s
TEST_FILE_WAV_TWO_MIN = AUDIO_FORMATS_DIR / "soundscape.wav"
TEST_FILE_WAV = TEST_FILE_WAV_TWO_MIN
NON_EXISTING_TEST_FILE_WAV = TEST_FILES_DIR / "dummy.wav"
TEST_FILE_MEAN_TF_FP32 = numpy.float32(0.06233002)
