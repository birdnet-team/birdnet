import os
import tempfile
from pathlib import Path

from birdnet.utils import download_file_tqdm


def test_download_model_v2p4_to_tmp():
  dl_path = "https://tuc.cloud/index.php/s/45KmTcpHH8iDDA2/download/BirdNET_v2.4.zip"
  output_path = Path(tempfile.gettempdir()) / "dl.zip"
  download_file_tqdm(dl_path, output_path)
  assert output_path.is_file()
  os.remove(output_path)
  assert not output_path.is_file()
