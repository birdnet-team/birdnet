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
  # The library resolves its own safe start method (see
  # birdnet.core.start_method.resolve_start_method): "spawn" unless the
  # application explicitly chose otherwise. This fixture resets the global to
  # "spawn" before each test for isolation: a start-method test that calls
  # set_start_method("fork"/"forkserver") would otherwise leak its choice into
  # the next test sharing the same xdist worker -- and because the resolver
  # honors an explicitly fixed global, a leaked "fork" would silently switch
  # every later pipeline test onto the deadlock-prone fork path. Tests that
  # need another method still override this via use_fork_or_skip() etc.; the
  # resolver's unset-global fallback is covered by unit tests
  # (core_py/test_start_method.py) since the global cannot be un-fixed here.
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
