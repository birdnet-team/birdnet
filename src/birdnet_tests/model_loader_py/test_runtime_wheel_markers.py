"""The wheel table in model_loader.py matches the pyproject.toml markers."""

import platform
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import birdnet.model_loader as ml

pytestmark = pytest.mark.no_tf

_PYPROJECT = Path(__file__).parents[3] / "pyproject.toml"
_MINORS = range(11, 30)  # far future: a marker must stop where its wheels stop
_ENVIRONMENTS = {
  "linux": [("linux", "x86_64"), ("linux", "aarch64")],
  "macos-arm64": [("darwin", "arm64")],
  "macos-x86_64": [("darwin", "x86_64")],
  "windows-amd64": [("win32", "AMD64")],
}


def _requirements(package: str, group: str) -> list[Requirement]:
  project = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]
  if group == "dependencies":
    specs = project["dependencies"]
  else:
    specs = project["optional-dependencies"][group]
  return [req for req in map(Requirement, specs) if req.name == package]


def _minors_installed(
  requirements: list[Requirement], sys_platform: str, machine: str
) -> list[int]:
  def applies(req: Requirement, minor: int) -> bool:
    environment = {
      "sys_platform": sys_platform,
      "platform_machine": machine,
      "python_version": f"3.{minor}",
      "python_full_version": f"3.{minor}.0",
    }
    return req.marker is None or req.marker.evaluate(environment)

  return [m for m in _MINORS if any(applies(req, m) for req in requirements)]


# Not macOS x86_64: the tf marker admits 3.13, which has no TensorFlow 2.16 wheel.
@pytest.mark.parametrize(
  ("package", "group", "platforms"),
  [
    ("tensorflow", "tf", ["linux", "macos-arm64", "windows-amd64"]),
    ("tensorflow", "and-cuda", ["linux"]),
    ("onnxruntime", "dependencies", list(_ENVIRONMENTS)),
    ("onnxruntime", "onnx", list(_ENVIRONMENTS)),
  ],
)
def test_wheel_table_matches_pyproject_markers(
  package: str, group: str, platforms: list[str]
) -> None:
  requirements = _requirements(package, group)
  assert requirements, f"no {package} requirement in {group}"

  for platform_key in platforms:
    expected = list(ml._WHEEL_MINORS[package][platform_key])
    for sys_platform, machine in _ENVIRONMENTS[platform_key]:
      installed = _minors_installed(requirements, sys_platform, machine)
      assert installed == expected, (platform_key, machine)


def test_litert_wheel_table_covers_the_base_dependency() -> None:
  # The marker stops at 3.13 though 3.14 has wheels (py314 installs it by hand).
  (requirement,) = _requirements("ai-edge-litert", "dependencies")

  for platform_key, environments in _ENVIRONMENTS.items():
    wheels = set(ml._WHEEL_MINORS["ai-edge-litert"][platform_key])
    for sys_platform, machine in environments:
      installed = set(_minors_installed([requirement], sys_platform, machine))
      assert installed <= wheels, (platform_key, machine)
      assert bool(installed) == bool(wheels), (platform_key, machine)


@pytest.mark.parametrize(
  ("platform_key", "sys_platform", "machine"),
  [
    (key, sys_platform, machine)
    for key, environments in _ENVIRONMENTS.items()
    for sys_platform, machine in environments
  ],
)
def test_wheel_platform_maps_each_marker_environment(
  monkeypatch: pytest.MonkeyPatch, platform_key: str, sys_platform: str, machine: str
) -> None:
  monkeypatch.setattr(sys, "platform", sys_platform)
  monkeypatch.setattr(platform, "machine", lambda: machine)

  assert ml._wheel_platform() == platform_key
