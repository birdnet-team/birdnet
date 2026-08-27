import hashlib
from pathlib import Path

import pytest

from birdnet.utils import helper
from birdnet.utils.helper import ModelInfo, ensure_single_file_model

_CONTENT = b"model bytes 0123456789"
_OTHER_CONTENT = b"other bytes 0123456789"
assert len(_OTHER_CONTENT) == len(_CONTENT)


def _model_info(content: bytes = _CONTENT) -> ModelInfo:
  return ModelInfo(
    dl_url="https://example.org/model.tflite",
    dl_size=len(content),
    file_size=len(content),
    dl_file_name="model.tflite",
    sha256=hashlib.sha256(content).hexdigest(),
  )


def _paths(tmp_path: Path, info: ModelInfo) -> tuple[Path, Path]:
  assert info.sha256 is not None
  model_path = tmp_path / f"model-fp32-{info.content_tag}.tflite"
  legacy_path = tmp_path / "model-fp32.tflite"
  return model_path, legacy_path


def _stub_download(monkeypatch: pytest.MonkeyPatch, served: bytes | None) -> list[str]:
  calls: list[str] = []

  def fake(
    url: str,
    file_path: Path,
    *,
    download_size: int | None = None,
    description: str | None = None,
  ) -> int:
    calls.append(url)
    assert served is not None, "no download was expected"
    file_path.write_bytes(served)
    return len(served)

  monkeypatch.setattr(helper, "download_file_tqdm", fake)
  return calls


@pytest.mark.no_tf
def test_downloads_when_nothing_is_cached(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  calls = _stub_download(monkeypatch, _CONTENT)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert calls == [info.dl_url]
  assert model_path.read_bytes() == _CONTENT


@pytest.mark.no_tf
def test_discards_a_download_whose_checksum_does_not_match(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Same length, different bytes: the size check alone would accept this."""
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  _stub_download(monkeypatch, _OTHER_CONTENT)

  with pytest.raises(RuntimeError, match="does not match its expected checksum"):
    ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert not model_path.exists()
  assert list(tmp_path.glob("*.unverified")) == []


@pytest.mark.no_tf
def test_skips_when_the_file_is_already_there(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  model_path.write_bytes(_CONTENT)
  calls = _stub_download(monkeypatch, None)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert calls == []


@pytest.mark.no_tf
def test_does_not_hash_on_a_warm_cache(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Hashing a ~540 MB model on every load would be a serious regression."""
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  model_path.write_bytes(_CONTENT)
  _stub_download(monkeypatch, None)

  def never(path: Path) -> str:
    raise AssertionError(f"hashed {path} on a warm cache")

  monkeypatch.setattr(helper, "sha256_file", never)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")


@pytest.mark.no_tf
def test_replaces_a_cached_file_of_the_wrong_size(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  model_path.write_bytes(b"truncated")
  calls = _stub_download(monkeypatch, _CONTENT)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert calls == [info.dl_url]
  assert model_path.read_bytes() == _CONTENT


@pytest.mark.no_tf
def test_adopts_a_matching_legacy_file_without_downloading(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  legacy_path.write_bytes(_CONTENT)
  calls = _stub_download(monkeypatch, None)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert calls == []
  assert model_path.read_bytes() == _CONTENT
  assert not legacy_path.exists()


@pytest.mark.no_tf
def test_skips_the_download_when_a_concurrent_process_adopted_first(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """The loser of the adoption race must use the winner's file, not re-fetch it."""
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  legacy_path.write_bytes(_CONTENT)
  calls = _stub_download(monkeypatch, None)

  def winner_replace(src: str | Path, dst: str | Path) -> None:
    Path(dst).write_bytes(_CONTENT)
    raise PermissionError("already being renamed by another process")

  monkeypatch.setattr(helper.os, "replace", winner_replace)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert calls == []
  assert model_path.read_bytes() == _CONTENT


@pytest.mark.no_tf
def test_survives_a_legacy_file_vanishing_mid_probe(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """A concurrent process renaming the legacy file away must not crash the load."""
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  legacy_path.write_bytes(_CONTENT)
  calls = _stub_download(monkeypatch, _CONTENT)

  real_sha256_file = helper.sha256_file

  def vanished(path: Path) -> str:
    if path == legacy_path:
      raise FileNotFoundError(path)
    return real_sha256_file(path)

  monkeypatch.setattr(helper, "sha256_file", vanished)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert calls == [info.dl_url]
  assert model_path.read_bytes() == _CONTENT


@pytest.mark.no_tf
def test_leaves_a_differing_legacy_file_to_its_owner(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Right size, other content: another installed version may still read it."""
  info = _model_info()
  model_path, legacy_path = _paths(tmp_path, info)
  legacy_path.write_bytes(_OTHER_CONTENT)
  calls = _stub_download(monkeypatch, _CONTENT)

  ensure_single_file_model(info, model_path, legacy_path, "downloading")

  assert calls == [info.dl_url]
  assert model_path.read_bytes() == _CONTENT
  assert legacy_path.read_bytes() == _OTHER_CONTENT
