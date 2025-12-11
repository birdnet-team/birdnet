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

"""


def get_package_logger() -> logging.Logger:
  return logging.getLogger(PKG_NAME)


# # The worker configuration is done at the start of the worker process run.
# # Note that on Windows you can't rely on fork semantics, so each process
# # will run the logging configuration code when it starts.
# def process_logging_configurer(logging_queue: Queue):
#   root = logging.getLogger()
#   assert root.level == logging.WARNING
#   assert root.hasHandlers() is False
#   h = QueueHandler(logging_queue)  # Just the one handler needed
#   root.setLevel(logging.NOTSET)
#   root.addHandler(h)


# def xadd_queue_handler(logging_queue: Queue) -> QueueHandler:
#   root = get_package_logger()
#   h = QueueHandler(logging_queue)  # Just the one handler needed
#   root.addHandler(h)
#   return h


# def xqueue_handler_exists(logging_queue: Queue) -> bool:
#   root = get_package_logger()
#   for handler in root.handlers:
#     if isinstance(handler, QueueHandler) and handler.queue is logging_queue:
#       return True
#   return False


# def xremove_queue_handler(handler: QueueHandler) -> None:
#   root = get_package_logger()
#   # check has queue handler already
#   assert handler in root.handlers
#   root.removeHandler(handler)


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
