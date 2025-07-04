import tempfile
from pathlib import Path

from birdnet_v1.models.v2m4.model_v2m4_tflite import DOWNLOAD_SIZE, DOWNLOAD_URL
from birdnet.utils import download_file_tqdm


def test_download_model_v2m4_to_tmp():
  with tempfile.TemporaryDirectory(prefix="birdnet.test_download_model_v2m4_to_tmp.") as tmp_dir:
    output_path = Path(tmp_dir) / "dl.zip"
    download_file_tqdm(
      DOWNLOAD_URL,
      output_path,
      download_size=DOWNLOAD_SIZE,
      description="Downloading model",
    )
    assert output_path.is_file()
