from pathlib import Path

import pytest

from birdnet.utils import helper
from birdnet.utils.label_manifest import LabelInput, ensure_artifact, sha256_bytes

_CONTENT = b"the pinned release\n"


@pytest.mark.no_tf
def test_interrupted_download_leaves_nothing_at_the_path(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Only verified bytes may land at the path, so half a file must not."""
  expected = LabelInput(
    path=tmp_path / "cache" / "artifact.csv",
    url="https://example.org/artifact.csv",
    size=len(_CONTENT),
    sha256=sha256_bytes(_CONTENT),
  )

  def cut_off(url: str, file_path: Path, **kwargs: object) -> int:
    file_path.write_bytes(_CONTENT[: len(_CONTENT) // 2])
    raise ConnectionError("connection reset mid-download")

  monkeypatch.setattr(helper, "download_file_tqdm", cut_off)

  with pytest.raises(ConnectionError):
    ensure_artifact(expected, "test artifact")

  assert not expected.path.exists()
  assert list(expected.path.parent.glob("*.unverified")) == []
