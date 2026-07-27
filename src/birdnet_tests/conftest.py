import logging
from multiprocessing import set_start_method

import pytest

from birdnet.utils.logging_utils import get_package_logger

# The v3.0 models are ~520 MiB and Zenodo serves them at roughly 2 MiB/s, so a single
# download takes ~5-6 min. That sits right on the global 300s timeout, which cut the
# downloads off just short of completion instead of letting them finish.
LOAD_MODEL_TIMEOUT_S = 1800


@pytest.fixture(autouse=True)
def _default_spawn_start_method() -> None:
  # The inference pipeline creates its processes with the global default start
  # method. On Linux that default is "fork", so a pipeline test that forks after
  # TensorFlow's multi-threaded runtime is loaded can deadlock in the child (the
  # classic fork-after-threads trap) and hang the whole run until the job timeout.
  # macOS/Windows already default to "spawn" and don't hit this. Force "spawn" per
  # test so every platform matches, and reset it before each test so a start-method
  # test that calls set_start_method("fork"/"forkserver") cannot leak its choice into
  # the next test sharing the same xdist worker. Tests that specifically need another
  # method still override this via use_fork_or_skip()/use_spawn_or_skip().
  set_start_method("spawn", force=True)


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
