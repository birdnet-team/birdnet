import os
from pathlib import Path

import numpy as np

# flag for "can be written to"
WRITE_FLAG = np.uint8(0)

# flag for "can be read from"
READ_FLAG = np.uint8(1)

# flag for "busy", i.e., currently being processed
BUSY_FLAG = np.uint8(2)

# flag for "done"
DONE_FLAG = np.uint8(3)


def get_app_data_path() -> Path:
  """Returns the appropriate application data path based on the operating system."""
  app_data_path: str
  if os.name == "nt":  # Windows
    app_data_path = os.getenv("APPDATA")
    assert app_data_path is not None
  elif os.name == "posix":
    if os.uname().sysname == "Darwin":  # Mac OS X
      app_data_path = os.path.expanduser("~/Library/Application Support")
    else:  # Linux
      app_data_path = os.path.expanduser("~/.local/share")
  else:
    raise OSError("Unsupported operating system")

  result = Path(app_data_path)
  return result


def get_birdnet_app_data_folder() -> Path:
  app_data = get_app_data_path()
  result = app_data / "birdnet"
  return result


APP_DIR = get_birdnet_app_data_folder()

if not APP_DIR.exists():
  APP_DIR.mkdir(parents=True, exist_ok=True)
