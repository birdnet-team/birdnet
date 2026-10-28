import pytest

from birdnet.core.backends import check_is_intel_macos


def test_check_is_intel_macos_false_off_mac(monkeypatch: pytest.MonkeyPatch) -> None:
  import platform

  monkeypatch.setattr(platform, "system", lambda: "Linux")
  assert check_is_intel_macos() is False


def test_check_is_intel_macos_true_on_intel_mac(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import platform

  monkeypatch.setattr(platform, "system", lambda: "Darwin")
  monkeypatch.setattr(platform, "machine", lambda: "x86_64")
  assert check_is_intel_macos() is True


def test_check_is_intel_macos_false_on_arm_mac(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import platform

  monkeypatch.setattr(platform, "system", lambda: "Darwin")
  monkeypatch.setattr(platform, "machine", lambda: "arm64")
  assert check_is_intel_macos() is False
