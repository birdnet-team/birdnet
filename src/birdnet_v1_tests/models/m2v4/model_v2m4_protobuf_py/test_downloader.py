import tempfile
from pathlib import Path

from birdnet_v1.models.v2m4.model_v2m4_protobuf import DOWNLOAD_URL, DownloaderProtobuf
from birdnet_v1.utils import download_file_tqdm


def get_dl_size() -> None:
  with tempfile.NamedTemporaryFile(prefix="birdnet.test_downloader.") as tmp_path:
    dl_size = download_file_tqdm(DOWNLOAD_URL, Path(tmp_path.name),
                                 description="Downloading models")
  print(dl_size)


def test_download_creates_all_files():
  with tempfile.TemporaryDirectory(prefix="birdnet.test_downloader.") as tmp_dir:
    downloader = DownloaderProtobuf(Path(tmp_dir))
    # pylint: disable=W0212:protected-access
    assert not downloader._check_model_files_exist()
    downloader.ensure_model_is_available()
    assert downloader._check_model_files_exist()


if __name__ == "__main__":
  get_dl_size()
