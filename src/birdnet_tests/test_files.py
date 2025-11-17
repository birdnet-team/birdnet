from pathlib import Path


TEST_RESULTS_DIR = Path("src/birdnet_tests/test_results")
TEST_FILES_DIR = Path("src/birdnet_tests/TEST_FILES")
AUDIO_FORMATS_DIR = TEST_FILES_DIR / "audio_formats"

# Duration: 120s
TEST_FILE_LONG = AUDIO_FORMATS_DIR / "soundscape.wav"
TEST_FILE_SHORT = AUDIO_FORMATS_DIR / "soundscape_7s.flac"  # 7.3 seconds
TEST_FILE_SHORT_SCORE_SHAPE = (1, 3, 6522)
TEST_FILE_SHORT_EMB_SHAPE = (1, 3, 1024)

NON_EXISTING_TEST_FILE_WAV = TEST_FILES_DIR / "dummy.wav"
