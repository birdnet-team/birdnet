from pathlib import Path

import pytest

from birdnet_tests.fake_acoustic_backend import fake_encode_session


def _callback(stats: object) -> None:
  pass


@pytest.mark.no_tf
@pytest.mark.parametrize("show_stats", [None, "minimal"])
def test_progress_callback_without_progress_stats_is_rejected(
  tmp_path: Path, show_stats: str | None
) -> None:
  with pytest.raises(ValueError, match=r"Progress callback can only be used"):
    fake_encode_session(
      tmp_path,
      show_stats=show_stats,  # type: ignore[arg-type]
      progress_callback=_callback,
    )


@pytest.mark.no_tf
@pytest.mark.parametrize("show_stats", ["progress", "benchmark"])
def test_progress_callback_with_progress_stats_is_accepted(
  tmp_path: Path, show_stats: str
) -> None:
  fake_encode_session(
    tmp_path,
    show_stats=show_stats,  # type: ignore[arg-type]
    progress_callback=_callback,
  )
