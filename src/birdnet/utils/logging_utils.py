# birdnet/logging_utils.py
from __future__ import annotations

import logging

from birdnet.globals import PKG_NAME

"""
loggers:

root:
- birdnet (INFO)
  - birdnet.session_XXX (INFO, inherited)
    - logger for each predict/encode session
    - birdnet.session_XXX.modules... e.g. birdnet.session_XXX.birdnet.acoustic_models.inference_pipeline.processes
- birdnet_file_writer.session_XXX (INFO, inherited)
  - file writer for each predict/encode session

"""  # noqa: E501


def get_package_logger() -> logging.Logger:
  return logging.getLogger(PKG_NAME)


def get_logger_for_package(name: str) -> logging.Logger:
  logger = logging.getLogger(name)
  logger.parent = get_package_logger()
  return logger


def get_package_logging_level() -> int:
  result = get_package_logger().level
  return result


def init_package_logger(logging_level: int) -> None:
  root = get_package_logger()
  root.setLevel(logging_level)
  root.propagate = False
