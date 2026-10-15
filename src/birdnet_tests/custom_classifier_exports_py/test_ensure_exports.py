"""`ensure_exports`: what it builds, and when it takes the build lock."""

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from birdnet_tests import custom_classifier_exports as cce
from birdnet_tests.custom_classifier_exports import ExportSpec

pytestmark = pytest.mark.no_tf

V2_4 = [ExportSpec("2.4", "replace", 0), ExportSpec("2.4", "append", 0)]
V3_0 = [ExportSpec("3.0", "replace", 0), ExportSpec("3.0", "replace", 0, "raven")]


class Calls:
  def __init__(self) -> None:
    self.locks: list[Path] = []
    self.builds: list[tuple[Path, list[ExportSpec]]] = []


def _place(cache_dir: Path, spec: ExportSpec) -> None:
  spec_dir = cache_dir / spec.name
  spec_dir.mkdir(parents=True)
  (spec_dir / "labels.txt").write_text("a\nb\nc\n", encoding="utf-8")


@pytest.fixture
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Calls:
  """Stub the official models, the lock and the build; keep the control flow."""
  cache_root = tmp_path / "exports"
  recorded = Calls()

  @contextlib.contextmanager
  def fake_lock(
    lock_dir: Path, description: str, timeout_s: float = 0
  ) -> Iterator[None]:
    recorded.locks.append(lock_dir)
    yield

  def fake_build(cache_dir: Path, specs: Sequence[ExportSpec]) -> None:
    recorded.builds.append((cache_dir, list(specs)))
    for spec in specs:
      _place(cache_dir, spec)

  monkeypatch.setattr(cce, "_CACHE_ROOT", cache_root)
  monkeypatch.setattr(cce, "_BUILD_LOCK", cache_root / "build.lock")
  monkeypatch.setattr(cce, "_can_build", lambda: True)
  monkeypatch.setattr(cce, "_official_pb", lambda version: (tmp_path, tmp_path))
  monkeypatch.setattr(cce, "_official_pb_on_disk_or_skip", lambda version: tmp_path)
  monkeypatch.setattr(
    cce, "_cache_dir", lambda version, pb_dir: cache_root / f"v{version}"
  )
  monkeypatch.setattr(cce, "directory_lock", fake_lock)
  monkeypatch.setattr(cce, "_build_in_subprocess", fake_build)
  return recorded


def test_cached_exports_are_returned_without_the_lock(calls: Calls) -> None:
  for spec in V2_4:
    _place(cce._CACHE_ROOT / "v2.4", spec)

  exports = cce.ensure_exports(V2_4)

  assert calls.locks == []
  assert calls.builds == []
  assert {spec: export.n_species for spec, export in exports.items()} == {
    V2_4[0]: 3,
    V2_4[1]: 3,
  }


def test_only_the_missing_exports_are_built_under_the_lock(calls: Calls) -> None:
  _place(cce._CACHE_ROOT / "v2.4", V2_4[0])

  exports = cce.ensure_exports(V2_4)

  assert calls.locks == [cce._BUILD_LOCK]
  assert calls.builds == [(cce._CACHE_ROOT / "v2.4", [V2_4[1]])]
  assert set(exports) == set(V2_4)


def test_both_bases_build_under_the_same_lock(calls: Calls) -> None:
  cce.ensure_exports([*V2_4, *V3_0])

  assert calls.locks == [cce._BUILD_LOCK, cce._BUILD_LOCK]
  assert [specs for _, specs in calls.builds] == [V2_4, V3_0]


def test_without_tensorflow_missing_exports_skip(
  calls: Calls, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(cce, "_can_build", lambda: False)
  _place(cce._CACHE_ROOT / "v2.4", V2_4[0])

  with pytest.raises(pytest.skip.Exception, match=r"not built yet"):
    cce.ensure_exports(V2_4)
  assert calls.locks == []
  assert calls.builds == []

  _place(cce._CACHE_ROOT / "v2.4", V2_4[1])
  assert set(cce.ensure_exports(V2_4)) == set(V2_4)
  assert calls.locks == []
