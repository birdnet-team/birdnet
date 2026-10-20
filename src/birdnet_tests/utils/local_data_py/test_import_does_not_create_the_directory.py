"""Importing the package does not create the app-data directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from birdnet.globals import ENV_VAR_APP_DATA

pytestmark = pytest.mark.no_tf


def test_import_leaves_a_missing_app_data_directory_missing(tmp_path: Path) -> None:
  app_data = tmp_path / "app-data"
  env = dict(os.environ, **{ENV_VAR_APP_DATA: str(app_data)})

  subprocess.run(
    [sys.executable, "-c", "import birdnet.utils.local_data"],
    env=env,
    check=True,
    timeout=120,
  )

  assert not app_data.exists()
