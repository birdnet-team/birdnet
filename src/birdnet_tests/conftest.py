import faulthandler
import logging
import os
from multiprocessing import set_start_method
from pathlib import Path
from typing import IO

import pytest

from birdnet.core.backends import tf_installed
from birdnet.utils.logging_utils import get_package_logger

# The v3.0 models are ~520 MiB and Zenodo serves them at roughly 2 MiB/s, so a single
# download takes ~5-6 min. That sat right on the old global 300s timeout, which cut
# the downloads off just short of completion instead of letting them finish.
LOAD_MODEL_TIMEOUT_S = 1800

# --- GIL-immune hang watchdog -------------------------------------------------
# pytest-timeout's thread method needs the GIL to dump stacks and kill the
# process, so a test that wedges native code while a thread holds the GIL (seen
# with fork-after-TensorFlow and a TF 2.20 eager-execution deadlock) survives it
# and burns the whole CI job budget in silence. faulthandler's watchdog is
# implemented in C: it dumps every thread's Python stack and hard-exits without
# ever taking the GIL. Armed per test at the test's own timeout plus a margin,
# so pytest-timeout always gets the first (friendlier) shot; between tests a
# long idle deadline covers wedges outside any test.
#
# The watchdog runs in every process: xdist workers, single-process runs AND
# the xdist controller. The controller matters: a worker that dies from a hard
# native crash (observed with a fork-lane worker on macOS: instant death, no
# Python traceback) can leave the controller waiting on the dead node forever
# with zero output. The controller re-arms on the logstart/logfinish events the
# workers forward, so when those stop, it dumps its own stacks and exits at the
# idle deadline instead of burning the job budget. Per-test deadlines come from
# a nodeid -> timeout map built at collection, because the logstart hook only
# receives the nodeid. faulthandler.enable() additionally catches hard crashes
# (SIGSEGV/SIGABRT/SIGBUS) with a stack in the same dump file.
#
# Opt-in via BIRDNET_TEST_WATCHDOG_DIR (set in CI, where the dump files are
# uploaded as artifacts on failure) because a hard kill would be hostile to
# local debugging sessions.
ENV_VAR_WATCHDOG_DIR = "BIRDNET_TEST_WATCHDOG_DIR"
WATCHDOG_MARGIN_S = 60
WATCHDOG_IDLE_S = 900

_watchdog_file: IO[str] | None = None
_watchdog_test_timeouts: dict[str, float] = {}
# Deadline used when a nodeid is missing from the map. Under xdist only the
# workers collect, so the controller's map is always empty and every logstart
# falls back to this value. It must therefore cover the longest legitimate
# test (a cold-cache load_model download, up to LOAD_MODEL_TIMEOUT_S), or the
# controller would kill a healthy run whose workers are all quietly
# downloading. pytest_configure lowers it to WATCHDOG_IDLE_S in processes
# that collect themselves (workers, -n 0 runs), where the map hit is the
# normal case and a miss means something is off.
_watchdog_fallback_s: float = LOAD_MODEL_TIMEOUT_S


def _watchdog_arm(seconds: float) -> None:
  if _watchdog_file is not None:
    faulthandler.dump_traceback_later(seconds, exit=True, file=_watchdog_file)


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


def _effective_test_timeout(item: pytest.Item) -> float:
  marker = item.get_closest_marker("timeout")
  if marker is not None and marker.args:
    return float(marker.args[0])
  return float(item.config.getini("timeout") or WATCHDOG_IDLE_S)


def _runs_without_tensorflow(item: pytest.Item) -> bool:
  # TensorFlow is an optional dependency, and most of the suite needs it. Without
  # it only the declared TensorFlow-free surface runs: `no_tf`, plus the litert
  # lane except the tests marked `tf`. Everything else is skipped, not failed.
  if item.get_closest_marker("tf") is not None:
    return False
  return (
    item.get_closest_marker("no_tf") is not None
    or item.get_closest_marker("litert") is not None
  )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
  tf_available = tf_installed()
  skip_needs_tf = pytest.mark.skip(
    reason="outside the TensorFlow-free test surface (no_tf/litert) and TensorFlow "
    "is not installed (pip install birdnet[tf])"
  )
  for item in items:
    if item.get_closest_marker("load_model") is not None:
      item.add_marker(pytest.mark.timeout(LOAD_MODEL_TIMEOUT_S))
    if not tf_available and not _runs_without_tensorflow(item):
      item.add_marker(skip_needs_tf)
    # Collection runs in workers and controller alike, so both can look up the
    # per-test deadline from the nodeid the logstart hook hands them.
    _watchdog_test_timeouts[item.nodeid] = _effective_test_timeout(item)


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
  if _watchdog_file is not None:
    _watchdog_file.write(f"===== STARTING: {nodeid} =====\n")
    _watchdog_file.flush()
    timeout = _watchdog_test_timeouts.get(nodeid, _watchdog_fallback_s)
    _watchdog_arm(timeout + WATCHDOG_MARGIN_S)


def pytest_runtest_logfinish(nodeid: str, location: tuple) -> None:
  _watchdog_arm(WATCHDOG_IDLE_S)


def pytest_unconfigure() -> None:
  global _watchdog_file
  if _watchdog_file is not None:
    faulthandler.cancel_dump_traceback_later()
    _watchdog_file.close()
    _watchdog_file = None


def pytest_configure(config: pytest.Config) -> None:
  global _watchdog_file, _watchdog_fallback_s
  is_worker = hasattr(config, "workerinput")
  is_xdist_controller = not is_worker and bool(config.getoption("numprocesses", None))
  if not is_xdist_controller:
    _watchdog_fallback_s = WATCHDOG_IDLE_S
  watchdog_dir = os.environ.get(ENV_VAR_WATCHDOG_DIR)
  if watchdog_dir:
    dump_dir = Path(watchdog_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    role = "worker" if is_worker else "main"
    _watchdog_file = open(  # noqa: SIM115 (closed in pytest_unconfigure)
      dump_dir / f"watchdog_{role}_{os.getpid()}.txt", "w", encoding="utf-8"
    )
    # Also catch hard crashes (SIGSEGV/SIGABRT/SIGBUS) with a stack dump;
    # dump_traceback_later alone only covers hangs.
    faulthandler.enable(file=_watchdog_file)
    _watchdog_arm(WATCHDOG_IDLE_S)

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
