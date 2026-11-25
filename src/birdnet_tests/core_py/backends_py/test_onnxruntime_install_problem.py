"""A half-removed ONNX Runtime, found in a fake site-packages.

Uninstalling one of two builds that share the folder 'onnxruntime' deletes the
files both shipped: it leaves the other's own libraries behind, or no folder at
all where the other had no files of its own.
"""

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

import pytest

from birdnet.core.backends import (
  _onnxruntime_builds,
  _onnxruntime_reinstall_advice,
  onnxruntime_install_problem,
)

pytestmark = pytest.mark.no_tf

_GPU_LEFTOVER = "onnxruntime/capi/libonnxruntime_providers_cuda.so"


def _use_site(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  package_files: list[str],
  builds: list[tuple[str, str, str]],
) -> Path:
  """Make a site-packages with these files and (name, version, top level) builds
  the only place imports and metadata are looked up."""
  site = tmp_path / "site-packages"
  site.mkdir()
  for relative in package_files:
    (site / relative).parent.mkdir(parents=True, exist_ok=True)
    (site / relative).write_bytes(b"")
  for name, version, top_level in builds:
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
      f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    (info / "top_level.txt").write_text(f"{top_level}\n")
  monkeypatch.setattr(sys, "path", [str(site)])
  monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
  return site


def test_a_half_removed_install_is_named_with_its_repair(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  site = _use_site(
    monkeypatch,
    tmp_path,
    [_GPU_LEFTOVER],
    [("onnxruntime-gpu", "1.26.0", "onnxruntime")],
  )

  problem = onnxruntime_install_problem()

  assert problem is not None
  assert (
    f"the folder '{site / 'onnxruntime'}' holds only part of ONNX Runtime" in problem
  )
  assert "although onnxruntime-gpu 1.26.0 is recorded as installed" in problem
  assert problem.endswith(
    "'pip uninstall -y onnxruntime onnxruntime-gpu && "
    "pip install onnxruntime-gpu==1.26.0'."
  )


def test_a_half_removed_install_reinstalls_the_gpu_build_over_the_cpu_build(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(
    monkeypatch,
    tmp_path,
    [_GPU_LEFTOVER],
    [
      ("onnxruntime", "1.32.0", "onnxruntime"),
      ("onnxruntime-gpu", "1.26.0", "onnxruntime"),
    ],
  )

  problem = onnxruntime_install_problem()

  assert problem is not None
  assert problem.endswith("&& pip install onnxruntime-gpu==1.26.0'.")


def test_a_build_recorded_without_its_folder_is_named_with_its_repair(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(monkeypatch, tmp_path, [], [("onnxruntime", "1.26.0", "onnxruntime")])

  problem = onnxruntime_install_problem()

  assert problem is not None
  assert problem.startswith(
    "onnxruntime 1.26.0 is recorded as installed, but the folder 'onnxruntime' "
    "is missing."
  )
  assert problem.endswith(
    "'pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime==1.26.0'."
  )


def test_a_half_removed_install_without_a_recorded_build_is_named(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(monkeypatch, tmp_path, [_GPU_LEFTOVER], [])

  problem = onnxruntime_install_problem()

  assert problem is not None
  assert "and no ONNX Runtime build is recorded as installed" in problem
  assert problem.endswith(
    "'pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime'."
  )


def test_a_half_removed_install_imported_before_the_check_is_named(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(monkeypatch, tmp_path, [_GPU_LEFTOVER], [])
  spec = importlib.util.find_spec("onnxruntime")
  assert spec is not None
  monkeypatch.setitem(sys.modules, "onnxruntime", importlib.util.module_from_spec(spec))

  assert onnxruntime_install_problem() is not None


def test_a_complete_install_has_no_problem(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(
    monkeypatch,
    tmp_path,
    ["onnxruntime/__init__.py", _GPU_LEFTOVER],
    [
      ("onnxruntime", "1.32.0", "onnxruntime"),
      ("onnxruntime-gpu", "1.26.0", "onnxruntime"),
    ],
  )

  assert onnxruntime_install_problem() is None


def test_a_missing_install_has_no_problem(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(monkeypatch, tmp_path, [], [])

  assert onnxruntime_install_problem() is None


def test_builds_are_the_distributions_that_ship_the_package(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(
    monkeypatch,
    tmp_path,
    ["onnxruntime/__init__.py"],
    [
      ("onnxruntime_gpu", "1.26.0", "onnxruntime"),
      ("onnxruntime", "1.32.0", "onnxruntime"),
      ("onnxruntime-tools", "1.7.0", "onnxruntime_tools"),
    ],
  )

  assert _onnxruntime_builds() == [
    ("onnxruntime", "1.32.0"),
    ("onnxruntime-gpu", "1.26.0"),
  ]


def test_builds_are_empty_where_the_metadata_cannot_be_read(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def unreadable() -> dict[str, list[str]]:
    raise OSError("unreadable dist-info")

  monkeypatch.setattr(importlib.metadata, "packages_distributions", unreadable)

  assert _onnxruntime_builds() == []


def test_builds_skip_a_distribution_without_a_name(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  _use_site(monkeypatch, tmp_path, [], [("onnxruntime", "1.32.0", "onnxruntime")])
  # Python 3.11 maps a dist-info without METADATA to the name None.
  monkeypatch.setattr(
    importlib.metadata,
    "packages_distributions",
    lambda: {"onnxruntime": [None, "onnxruntime"]},
  )

  assert _onnxruntime_builds() == [("onnxruntime", "1.32.0")]


_CPU = ("onnxruntime", "1.32.0")
_GPU = ("onnxruntime-gpu", "1.26.0")
_DML = ("onnxruntime-directml", "1.24.0")


@pytest.mark.parametrize(
  ("builds", "keep", "advice"),
  [
    (
      [_CPU, _GPU],
      [_GPU],
      "'pip uninstall -y onnxruntime onnxruntime-gpu && "
      "pip install onnxruntime-gpu==1.26.0'",
    ),
    (
      [_CPU],
      [_CPU],
      "'pip uninstall -y onnxruntime onnxruntime-gpu && "
      "pip install onnxruntime==1.32.0'",
    ),
    (
      [],
      [],
      "'pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime'",
    ),
    (
      [_DML, _GPU],
      [_DML, _GPU],
      "'pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu', then "
      "install one of onnxruntime-directml or onnxruntime-gpu",
    ),
  ],
)
def test_the_advice_removes_every_build_before_reinstalling_one(
  builds: list[tuple[str, str]], keep: list[tuple[str, str]], advice: str
) -> None:
  assert _onnxruntime_reinstall_advice(builds, keep) == advice
