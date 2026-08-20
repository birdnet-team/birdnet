from collections.abc import Callable
from pathlib import Path

import pytest

from birdnet.utils.label_manifest import LabelInput, sha256_bytes, write_manifest

GENERATOR = "test_generator"
GENERATION_VERSION = 1
LANGUAGES = {"en_us": "com_name", "de": "common_name_de"}

_TAXONOMY_BYTES = b"sci_name,com_name,common_name_de\nParus major,Great Tit,Kohlmeise\n"
_LABELS_BYTES = b"sci_name;com_name\nParus major;Great Tit\n"


def _write_input(path: Path, data: bytes, url: str) -> LabelInput:
  path.write_bytes(data)
  return LabelInput(path=path, url=url, size=len(data), sha256=sha256_bytes(data))


@pytest.fixture
def lang_dir(tmp_path: Path) -> Path:
  d = tmp_path / "labels"
  d.mkdir()
  return d


@pytest.fixture
def inputs(tmp_path: Path) -> dict[str, LabelInput]:
  return {
    "labels": _write_input(
      tmp_path / "labels_raw.csv", _LABELS_BYTES, "https://example.org/labels.csv"
    ),
    "taxonomy": _write_input(
      tmp_path / "taxonomy.csv", _TAXONOMY_BYTES, "https://example.org/taxonomy.csv"
    ),
  }


@pytest.fixture
def generate(lang_dir: Path, inputs: dict[str, LabelInput]) -> Callable[[], None]:
  """Write a label directory the way a generator would, manifest last."""

  def _generate() -> None:
    for lang in LANGUAGES:
      (lang_dir / f"{lang}.txt").write_text(f"Parus major_{lang}", encoding="utf-8")
    write_manifest(
      lang_dir,
      generator=GENERATOR,
      generator_version=GENERATION_VERSION,
      inputs=inputs,
      languages=LANGUAGES,
      lang_files=[lang_dir / f"{lang}.txt" for lang in LANGUAGES],
    )

  return _generate
