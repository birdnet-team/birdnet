from pathlib import Path

import pytest

from birdnet_tests.test_files import TEST_FILE_LONG, TEST_FILE_SHORT, TEST_FILES_DIR

pytestmark = pytest.mark.no_tf


def test_the_test_data_is_found_from_any_working_directory(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  assert TEST_FILES_DIR.is_absolute()
  monkeypatch.chdir(tmp_path)
  assert TEST_FILE_SHORT.is_file()
  assert TEST_FILE_LONG.is_file()
