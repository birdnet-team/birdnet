"""Build a queue that really holds a half-written message.

A `multiprocessing.Queue` writer loops on partial pipe writes, so a process
killed while the pipe is full is cut mid-frame. The recipe here makes that
deterministic instead of a race: the payload is far larger than any platform's
pipe capacity, so the child's feeder thread is *always* blocked inside the
frame when the kill lands, and the pipe is left holding a length header whose
body will never be completed.

POSIX-specific by nature: Windows uses message-mode pipes with no length
header, so the same kill loses whole messages but cannot tear one. Tests use
this to show behaviour on both platforms, asserting the tear only where it can
exist.

Lives in its own importable module because the child function must be
picklable by qualified name for the ``spawn`` start method.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing import Queue
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event

# Larger than any default pipe capacity (Linux 64 KiB, macOS up to 64 KiB,
# Windows named-pipe buffers less), so the first frame already overflows it and
# the writer is blocked mid-frame from the first message on.
_PAYLOAD_BYTES = 256 * 1024


def _flood_and_hold(q: Queue, primed: Event) -> None:
  # `put` only hands the payload to the feeder thread and returns, so all puts
  # complete instantly; the feeder then writes until the pipe is full and
  # blocks inside a frame. The event says the puts happened -- the parent still
  # waits a moment for the feeder to reach the blocked write.
  for _ in range(4):
    q.put(b"x" * _PAYLOAD_BYTES)
  primed.set()
  time.sleep(300)


def make_torn_queue() -> tuple[Queue, BaseProcess]:
  """A queue whose pipe holds a message no reader can ever finish (POSIX).

  Returns the queue and the already-dead writer process. The parent keeps its
  own handle on the queue, so the reader sees no EOF -- exactly the situation
  a killed pipeline child leaves behind.
  """
  ctx = mp.get_context("spawn")
  q = ctx.Queue()
  primed = ctx.Event()
  writer = ctx.Process(target=_flood_and_hold, args=(q, primed), daemon=True)
  writer.start()
  assert primed.wait(timeout=60), "the writer never primed the queue"
  # Wait until bytes are visible on the parent's end of the pipe: the frame is
  # far larger than the pipe, so once any of it has arrived the feeder is
  # provably inside the frame and blocked. An observable condition, not a
  # sleep -- a loaded runner merely takes longer to reach it. (Private API,
  # but this helper exists precisely to manufacture the queue's failure mode.)
  deadline = time.monotonic() + 60
  while not q._reader.poll(0.1):  # type: ignore[attr-defined]
    assert time.monotonic() < deadline, "the feeder never reached the pipe"
  writer.kill()
  writer.join(timeout=30)
  assert not writer.is_alive(), "the writer did not die"
  return q, writer
