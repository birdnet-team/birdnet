import logging

import pytest

from birdnet.utils.logging_utils import get_package_logger

# The v3.0 models are ~520 MiB and Zenodo serves them at roughly 2 MiB/s, so a single
# download takes ~5-6 min. That sits right on the global 300s timeout, which cut the
# downloads off just short of completion instead of letting them finish.
LOAD_MODEL_TIMEOUT_S = 1800


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
  for item in items:
    if item.get_closest_marker("load_model") is not None:
      item.add_marker(pytest.mark.timeout(LOAD_MODEL_TIMEOUT_S))


def pytest_configure() -> None:
  loggers = {"tensorflow", "absl", "urllib3"}
  for l_name in loggers:
    logger = logging.getLogger(l_name)
    logger.disabled = True
    logger.propagate = False

  # gpus = tf.config.list_physical_devices('GPU')
  # for gpu in gpus:
  #   try:
  #     tf.config.experimental.set_memory_growth(gpu, True)
  #   except RuntimeError as e:
  #     print(e)

  main_logger = logging.getLogger()
  main_logger.setLevel(logging.DEBUG)
  main_logger.manager.disable = logging.NOTSET
  console = logging.StreamHandler()
  console.setLevel(logging.DEBUG)
  main_logger.addHandler(console)

  logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s (%(levelname)s): %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  root = get_package_logger()
  root.setLevel(logging.DEBUG)
