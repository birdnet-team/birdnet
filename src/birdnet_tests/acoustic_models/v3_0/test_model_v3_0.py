from pathlib import Path

import birdnet.acoustic.models.v3_0.model as model_module
from birdnet.acoustic.models.v3_0.model import AcousticDownloaderBaseV3_0
from birdnet.globals import VALID_MODEL_LANGUAGES_V3_0


class ProbeDownloader(AcousticDownloaderBaseV3_0):
  lang_dir: Path

  @classmethod
  def _get_lang_dir(cls) -> Path:
    return cls.lang_dir


def test_generate_lang_files_from_taxonomy(
  tmp_path, monkeypatch
) -> None:
  labels_raw = tmp_path / "labels_raw.csv"
  labels_raw.write_text(
    "sci_name;com_name\n"
    "Aaa aaa;English One\n"
    "Bbb bbb;English Two\n",
    encoding="utf-8",
  )
  taxonomy = tmp_path / "taxonomy.csv"
  taxonomy.write_text(
    "sci_name,com_name,common_name_de,common_name_zh-CN\n"
    "Aaa aaa,English One,Deutsch Eins,中文一\n",
    encoding="utf-8",
  )

  monkeypatch.setattr(model_module, "_LABELS_RAW_PATH", labels_raw)
  monkeypatch.setattr(model_module, "_TAXONOMY_PATH", taxonomy)
  ProbeDownloader.lang_dir = tmp_path / "labels"

  ProbeDownloader._generate_lang_files()

  assert list(ProbeDownloader.AVAILABLE_LANGUAGES) == VALID_MODEL_LANGUAGES_V3_0
  assert (ProbeDownloader.lang_dir / "en_us.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_English One\nBbb bbb_English Two"
  )
  assert (ProbeDownloader.lang_dir / "de.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_Deutsch Eins\nBbb bbb_English Two"
  )
  assert (ProbeDownloader.lang_dir / "zh.txt").read_text(encoding="utf-8") == (
    "Aaa aaa_中文一\nBbb bbb_English Two"
  )
