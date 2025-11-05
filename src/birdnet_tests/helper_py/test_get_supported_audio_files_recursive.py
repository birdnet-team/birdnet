from pathlib import Path
from tempfile import TemporaryDirectory

from birdnet.helper import get_supported_audio_files_recursive


def test_component_test() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    subfolder = folder / "subfolder"
    subfolder.mkdir()
    file_a = subfolder / "a.wav"
    file_a.touch()
    file_b = folder / "b.txt"
    file_b.touch()
    file_c = folder / "c.mp3"
    file_c.touch()
    file_d = subfolder / "d.docx"
    file_d.touch()
    file_e = subfolder / "subsubfolder" / "e.flac"
    file_e.parent.mkdir(parents=True)
    file_e.touch()
    result = list(get_supported_audio_files_recursive(folder))
    assert file_a.absolute() in result
    assert file_c.absolute() in result
    assert file_e.absolute() in result
    assert len(result) == 3


def test_empty_folder_returns_nothing() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    result = list(get_supported_audio_files_recursive(folder))
    assert result == []


def test_one_supported_file() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    file_a = folder / "a.wav"
    file_a.touch()
    result = list(get_supported_audio_files_recursive(folder))
    assert file_a.absolute() in result
    assert len(result) == 1


def test_one_supported_file_in_subfolder() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    subfolder = folder / "subfolder"
    subfolder.mkdir()
    file_a = subfolder / "a.wav"
    file_a.touch()
    result = list(get_supported_audio_files_recursive(folder))
    assert file_a.absolute() in result
    assert len(result) == 1


def test_one_supported_file_in_subsubfolder() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    subfolder = folder / "subfolder" / "subsubfolder"
    subfolder.mkdir(parents=True)
    file_a = subfolder / "a.wav"
    file_a.touch()
    result = list(get_supported_audio_files_recursive(folder))
    assert file_a.absolute() in result
    assert len(result) == 1


def test_two_supported_files() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    file_a = folder / "a.wav"
    file_b = folder / "b.wav"
    file_a.touch()
    file_b.touch()
    result = list(get_supported_audio_files_recursive(folder))
    assert file_a.absolute() in result
    assert file_b.absolute() in result
    assert len(result) == 2


def test_no_supported_file_returns_empty() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    file_a = folder / "a.txt"
    file_a.touch()
    result = list(get_supported_audio_files_recursive(folder))
    assert len(result) == 0
