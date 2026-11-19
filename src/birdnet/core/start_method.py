import multiprocessing as mp
import os

from birdnet.globals import ENV_VAR_START_METHOD


def resolve_start_method() -> str:
  """Resolve the multiprocessing start method for birdnet's pipeline processes.

  Resolution order:

  1. The ``BIRDNET_START_METHOD`` environment variable, if set (must be one of
     the platform's available methods).
  2. A start method the application has already fixed globally (e.g. via
     ``multiprocessing.set_start_method``) is honored unchanged.
  3. Otherwise ``spawn``.

  The fallback deliberately avoids ``fork``, which is the platform default on
  Linux: the inference pipeline is regularly started after TensorFlow has spun
  up its multi-threaded runtime, and forking a multi-threaded process can
  deadlock the child (CPython itself warns about this since 3.12). ``spawn``
  is already the platform default on macOS and Windows, so this only changes
  the implicit default on Linux. ``forkserver`` was rejected as the fallback
  because its default ``__main__`` preload imports the application's main
  module -- whose top-level ``import tensorflow`` runs regardless of any
  ``__main__`` guard -- into the fork server itself, reintroducing the
  fork-after-threads risk through the back door. Both ``fork`` and
  ``forkserver`` remain fully supported via explicit opt-in (rule 1 or 2);
  workers load their model themselves under every non-``fork`` method (lazy
  init), while ``fork`` gives opted-in applications copy-on-write model
  inheritance.

  Reading the global default uses ``allow_none=True`` on purpose: plain
  ``get_start_method()`` would *fix* the platform default as a side effect and
  thereby take the choice away from the application.
  """
  env_method = os.environ.get(ENV_VAR_START_METHOD)
  if env_method:
    if env_method not in mp.get_all_start_methods():
      raise ValueError(
        f"{ENV_VAR_START_METHOD}={env_method!r} is not supported on this "
        f"platform. Available methods: {mp.get_all_start_methods()}"
      )
    return env_method

  globally_fixed = mp.get_start_method(allow_none=True)
  if globally_fixed is not None:
    return globally_fixed

  return "spawn"


# Shared by the error a process raises while it imports the main script and by
# the parent's report of a process that died doing so.
MAIN_GUARD_HINT = (
  'the code that calls birdnet has to be under `if __name__ == "__main__":`, '
  "because every process birdnet starts imports the main script again"
)


def raise_if_importing_main(start_method: str) -> None:
  """Fail in a process that is still importing the main script.

  Under ``spawn`` and ``forkserver`` a new process imports the main script
  before it runs its target. Code there that calls birdnet outside of
  ``if __name__ == "__main__":`` runs again in that process, and multiprocessing
  refuses to start a process from it with either method. Raising before
  anything is set up names the cause and leaves nothing to tear down.

  `start_method` is the one the caller is about to use. ``fork`` is let through,
  because a fork during that import works: on Linux with Python 3.11 and 3.12 a
  spawned process sees ``fork`` as the fixed method while it imports.
  """
  if start_method == "fork":
    return
  # The flag multiprocessing sets around that import, and checks itself before
  # it refuses to start a process.
  if getattr(mp.current_process(), "_inheriting", False):
    raise RuntimeError(
      "birdnet was called while a process it started was importing the main "
      f"script: {MAIN_GUARD_HINT}."
    )
