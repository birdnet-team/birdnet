from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from birdnet.acoustic_models.inference_pipeline.configs import InferenceConfig


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
    with TemporaryDirectory() as tmpdir2:
      folder2 = Path(tmpdir2)
      file_f = folder2 / "f.wav"
      file_f.touch()
      result = InferenceConfig.validate_input_files([folder, file_c, file_f, file_f])
      assert result == sorted(
        [
          file_f.absolute(),
          file_c.absolute(),
          file_a.absolute(),
          file_e.absolute(),
        ]
      )


def test_one_supported_file() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    file_a = folder / "a.wav"
    file_a.touch()
    result = InferenceConfig.validate_input_files(folder)
    assert result == [file_a.absolute()]


def test_one_supported_file_as_str() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    file_a = folder / "a.wav"
    file_a.touch()
    result = InferenceConfig.validate_input_files(str(folder.absolute()))
    assert result == [file_a.absolute()]


def test_one_supported_file_in_subfolder() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    subfolder = folder / "subfolder"
    subfolder.mkdir()
    file_a = subfolder / "a.wav"
    file_a.touch()
    result = InferenceConfig.validate_input_files(folder)
    assert result == [file_a.absolute()]


def test_one_supported_file_in_subsubfolder() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    subfolder = folder / "subfolder" / "subsubfolder"
    subfolder.mkdir(parents=True)
    file_a = subfolder / "a.wav"
    file_a.touch()
    result = InferenceConfig.validate_input_files(folder)
    assert result == [file_a.absolute()]


def test_two_supported_files() -> None:
  with TemporaryDirectory() as tmpdir:
    folder = Path(tmpdir)
    file_a = folder / "a.wav"
    file_b = folder / "b.wav"
    file_a.touch()
    file_b.touch()
    result = InferenceConfig.validate_input_files(folder)
    assert result == [file_a.absolute(), file_b.absolute()]


def test_empty_folder_raise_error() -> None:
  with (
    pytest.raises(
      ValueError, match=r"No valid audio files were found in the provided input paths."
    ),
    TemporaryDirectory() as tmpdir,
  ):
    folder = Path(tmpdir)
    result = InferenceConfig.validate_input_files(folder)
    assert result == []


def test_no_supported_file_found_in_dir_raise_error() -> None:
  with (
    pytest.raises(
      ValueError, match=r"No valid audio files were found in the provided input paths."
    ),
    TemporaryDirectory() as tmpdir,
  ):
    folder = Path(tmpdir)
    file_a = folder / "a.txt"
    file_a.touch()
    InferenceConfig.validate_input_files(folder)


def test_no_supported_file_raise_error() -> None:
  with (
    pytest.raises(
      ValueError,
      match=r"Input file '.+' is not a supported audio format! Supported formats: .+",
    ),
    TemporaryDirectory() as tmpdir,
  ):
    folder = Path(tmpdir)
    file_a = folder / "a.txt"
    file_a.touch()
    InferenceConfig.validate_input_files(file_a)


def test_wrong_type_raise_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Unsupported input type: <class 'int'>",
  ):
    InferenceConfig.validate_input_files(123)  # type: ignore


def test_wrong_type_in_list_raise_error() -> None:
  with pytest.raises(
    ValueError,
    match=r"Unsupported input type: <class 'int'>",
  ):
    InferenceConfig.validate_input_files([123])  # type: ignore
