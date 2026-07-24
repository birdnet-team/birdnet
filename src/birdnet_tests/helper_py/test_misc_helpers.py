from pathlib import Path

import numpy as np
import pytest

from birdnet.utils.helper import (
  apply_speed_to_duration,
  apply_speed_to_samples,
  check_protobuf_model_files_exist,
  duration_as_samples,
  fillup_with_silence,
  flat_sigmoid_logaddexp_fast,
  format_input_for_csv,
  get_file_formats,
  get_hash,
  get_hop_duration_s,
  get_n_segments_speed,
  get_species_from_file,
  hms_centis_fast,
  is_supported_audio_file,
  itertools_batched,
  validate_species_list,
)


def test_hms_centis_fast() -> None:
  assert hms_centis_fast(0) == "00:00:00.00"
  assert hms_centis_fast(3661.5) == "01:01:01.50"
  assert hms_centis_fast(59.99) == "00:00:59.99"


def test_format_input_for_csv() -> None:
  assert format_input_for_csv("bird") == '"bird"'
  assert format_input_for_csv(42) == '"42"'


def test_get_hash_is_deterministic() -> None:
  assert get_hash("session") == get_hash("session")
  assert get_hash("session") != get_hash("other")
  # sha256 hex digest is 64 characters
  assert len(get_hash("session")) == 64


def test_apply_speed_to_duration() -> None:
  assert apply_speed_to_duration(10.0, 2.0) == 20.0
  assert apply_speed_to_duration(10.0, 0.5) == 5.0


def test_apply_speed_to_duration_rejects_non_positive_speed() -> None:
  with pytest.raises(AssertionError):
    apply_speed_to_duration(10.0, 0.0)


def test_apply_speed_to_samples_rounds() -> None:
  assert apply_speed_to_samples(10, 0.5) == 5
  assert apply_speed_to_samples(3, 0.5) == 2  # 1.5 rounds to nearest even (2)


def test_get_hop_duration_s() -> None:
  assert get_hop_duration_s(3.0, 1.0, 1.0) == 2.0
  assert get_hop_duration_s(3.0, 0.0, 2.0) == 6.0


def test_get_hop_duration_s_rejects_overlap_ge_segment() -> None:
  with pytest.raises(AssertionError):
    get_hop_duration_s(3.0, 3.0, 1.0)


def test_get_n_segments_speed() -> None:
  assert get_n_segments_speed(10.0, 3.0, 0.0, 1.0) == 4
  # a segment exactly filling the duration -> a single segment
  assert get_n_segments_speed(3.0, 3.0, 0.0, 1.0) == 1


def test_duration_as_samples() -> None:
  assert duration_as_samples(1.0, 48000) == 48000
  assert duration_as_samples(0.5, 48000) == 24000


def test_fillup_with_silence_pads() -> None:
  segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  result = fillup_with_silence(segment, 5)
  np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 0.0, 0.0])
  assert result.dtype == np.float32


def test_fillup_with_silence_returns_same_when_full() -> None:
  segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  result = fillup_with_silence(segment, 3)
  assert result is segment


def test_fillup_with_silence_rejects_too_short_target() -> None:
  segment = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  with pytest.raises(AssertionError):
    fillup_with_silence(segment, 2)


def test_flat_sigmoid_logaddexp_fast_zero_is_half() -> None:
  result = flat_sigmoid_logaddexp_fast(
    np.array([0.0], dtype=np.float32), sensitivity=1.0
  )
  np.testing.assert_allclose(result, [0.5], atol=1e-6)


def test_flat_sigmoid_logaddexp_fast_is_bounded_and_monotonic() -> None:
  # The prediction worker calls this with sensitivity=-1.0, which yields the
  # standard increasing sigmoid; clipping keeps extreme logits finite.
  x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0], dtype=np.float32)
  result = flat_sigmoid_logaddexp_fast(x, sensitivity=-1.0)
  assert np.all(result >= 0.0)
  assert np.all(result <= 1.0)
  assert np.all(np.isfinite(result))
  # sigmoid is monotonically increasing in the logit
  assert np.all(np.diff(result) >= 0)


def test_flat_sigmoid_logaddexp_fast_matches_reference_sigmoid() -> None:
  # With sensitivity=-1 and bias=1 (transformed_bias=0) it is 1 / (1 + e^-x).
  x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
  result = flat_sigmoid_logaddexp_fast(x, sensitivity=-1.0)
  expected = 1.0 / (1.0 + np.exp(-x))
  np.testing.assert_allclose(result, expected, atol=1e-9)


def test_itertools_batched() -> None:
  assert list(itertools_batched("ABCDEFG", 3)) == [
    ("A", "B", "C"),
    ("D", "E", "F"),
    ("G",),
  ]


def test_itertools_batched_empty() -> None:
  assert list(itertools_batched([], 3)) == []


def test_itertools_batched_rejects_non_positive_n() -> None:
  with pytest.raises(ValueError, match=r"n must be at least one"):
    list(itertools_batched("ABC", 0))


def test_get_file_formats() -> None:
  paths = {Path("a.WAV"), Path("b.mp3"), Path("c.wav")}
  assert get_file_formats(paths) == "MP3, WAV"


def test_is_supported_audio_file(tmp_path: Path) -> None:
  wav = tmp_path / "a.wav"
  wav.touch()
  txt = tmp_path / "b.txt"
  txt.touch()
  upper = tmp_path / "c.FLAC"
  upper.touch()

  assert is_supported_audio_file(wav)
  assert is_supported_audio_file(upper)
  assert not is_supported_audio_file(txt)


def test_get_species_from_file(tmp_path: Path) -> None:
  species_file = tmp_path / "species.txt"
  species_file.write_text("species_a\nspecies_b\nspecies_c\n", encoding="utf-8")
  species = get_species_from_file(species_file)
  assert list(species) == ["species_a", "species_b", "species_c"]


def test_validate_species_list(tmp_path: Path) -> None:
  species_file = tmp_path / "species.txt"
  species_file.write_text("species_a\nspecies_b\n", encoding="utf-8")
  species = validate_species_list(species_file)
  assert list(species) == ["species_a", "species_b"]


def test_validate_species_list_empty_raises_error(tmp_path: Path) -> None:
  species_file = tmp_path / "species.txt"
  species_file.write_text("", encoding="utf-8")
  with pytest.raises(ValueError, match=r"is empty!"):
    validate_species_list(species_file)


def test_validate_species_list_missing_file_raises_error(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match=r"Failed to read species list"):
    validate_species_list(tmp_path / "does_not_exist.txt")


def test_check_protobuf_model_files_exist(tmp_path: Path) -> None:
  assert not check_protobuf_model_files_exist(tmp_path)

  variables = tmp_path / "variables"
  variables.mkdir()
  (tmp_path / "saved_model.pb").write_bytes(b"pb")
  (variables / "variables.data-00000-of-00001").write_bytes(b"data")
  (variables / "variables.index").write_bytes(b"index")

  assert check_protobuf_model_files_exist(tmp_path)


def test_check_protobuf_model_files_exist_missing_index(tmp_path: Path) -> None:
  variables = tmp_path / "variables"
  variables.mkdir()
  (tmp_path / "saved_model.pb").write_bytes(b"pb")
  (variables / "variables.data-00000-of-00001").write_bytes(b"data")

  assert not check_protobuf_model_files_exist(tmp_path)
