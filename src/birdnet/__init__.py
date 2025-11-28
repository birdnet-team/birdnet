import logging

from birdnet.logging_utils import init_package_logger
from birdnet.model_loader import load, load_custom, load_perch_v2  # noqa: F401

init_package_logger(logging.INFO)
