from pathlib import Path

TEST_RESULTS_DIR = Path("src/birdnet_tests/test_results")
TEST_FILES_DIR = Path("src/birdnet_tests/TEST_FILES")
AUDIO_FORMATS_DIR = TEST_FILES_DIR / "audio_formats"

# Duration: 120s
TEST_FILE_WAV = AUDIO_FORMATS_DIR / "soundscape.wav"
NON_EXISTING_TEST_FILE_WAV = TEST_FILES_DIR / "dummy.wav"
