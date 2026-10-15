"""What names a build's cache directory, and what a failed build does."""

import subprocess
from pathlib import Path

import pytest

from birdnet_tests import custom_classifier_exports as cce
from birdnet_tests.custom_classifier_exports import ExportSpec

pytestmark = pytest.mark.no_tf


def _saved_model(directory: Path, weights: bytes) -> Path:
  (directory / "variables").mkdir(parents=True)
  (directory / "saved_model.pb").write_bytes(b"graph")
  (directory / "variables" / "variables.index").write_bytes(b"index")
  (directory / "variables" / "variables.data-00000-of-00001").write_bytes(weights)
  return directory


def test_the_key_follows_the_model_bytes(tmp_path: Path) -> None:
  one = _saved_model(tmp_path / "one", b"weights")
  same = _saved_model(tmp_path / "same", b"weights")
  other = _saved_model(tmp_path / "other", b"retrained")

  assert cce._cache_dir("2.4", one) == cce._cache_dir("2.4", same)
  assert cce._cache_dir("2.4", one) != cce._cache_dir("2.4", other)
  assert cce._cache_dir("2.4", one).name.startswith("v2.4-")


def test_the_key_follows_the_recipe(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  pb_dir = _saved_model(tmp_path / "pb", b"weights")
  before = cce._cache_dir("3.0", pb_dir)

  monkeypatch.setattr(cce, "_recipe_source", lambda: b"an edited recipe")

  assert cce._cache_dir("3.0", pb_dir) != before


def test_a_failed_build_raises_with_its_output(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  def failing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="boom")

  monkeypatch.setattr(cce.subprocess, "run", failing_run)

  with pytest.raises(RuntimeError, match=r"(?s)exit 1.*boom"):
    cce._build_in_subprocess(tmp_path, [ExportSpec("2.4", "replace", 0)])
