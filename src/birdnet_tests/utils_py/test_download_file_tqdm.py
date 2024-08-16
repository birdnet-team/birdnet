import tempfile
from pathlib import Path

from birdnet.utils import download_file_tqdm


def test_download_model_v2m4_to_tmp():
  dl_path = "https://tuc.cloud/index.php/s/45KmTcpHH8iDDA2/download/BirdNET_v2.4.zip"
  with tempfile.TemporaryDirectory(prefix="birdnet.test_download_model_v2m4_to_tmp.") as tmp_dir:
    output_path = Path(tmp_dir) / "dl.zip"
    download_file_tqdm(
      dl_path,
      output_path,
      download_size=76823623,
      description="Downloading model",
    )
    assert output_path.is_file()

