import logging
from birdnet.logging_utils import init_package_logger
from birdnet.model_loader import (  # noqa: F401
  load,
  load_custom,
)

init_package_logger(logging.INFO)
