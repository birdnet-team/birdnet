"""The conftest guard: an unmarked test cannot reach the internet for a model."""

from pathlib import Path

import pytest

from birdnet.utils.helper import download_file_tqdm

pytestmark = pytest.mark.no_tf


def test_unmarked_test_cannot_download(tmp_path: Path) -> None:
  with pytest.raises(pytest.fail.Exception, match="marked load_model"):
    download_file_tqdm(
      "https://zenodo.org/records/0/files/nothing.zip", tmp_path / "nothing.zip"
    )
