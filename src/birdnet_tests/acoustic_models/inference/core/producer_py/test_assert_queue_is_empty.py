from multiprocessing import Queue

from birdnet.acoustic.inference.core.producer import assert_queue_is_empty


def test_assert_queue_is_empty_passes_for_empty_queue() -> None:
  queue: Queue = Queue()
  # must not raise
  assert_queue_is_empty(queue)


def test_assert_queue_is_empty_raises_for_non_empty_queue() -> None:
  import time

  queue: Queue = Queue()
  queue.put(1)
  # A multiprocessing.Queue flushes items to the underlying pipe on a background
  # feeder thread, so the item only becomes visible after a short, non-
  # deterministic delay. Poll until the leftover item is detected.
  raised = False
  deadline = time.monotonic() + 5.0
  while time.monotonic() < deadline:
    try:
      assert_queue_is_empty(queue)
    except AssertionError:
      raised = True
      break
    time.sleep(0.01)
  assert raised, "assert_queue_is_empty did not detect the queued item"
