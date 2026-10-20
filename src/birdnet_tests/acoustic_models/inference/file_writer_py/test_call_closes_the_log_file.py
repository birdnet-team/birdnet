"""The log writer closes the session log file when it stops.

``MemoryHandler.close()`` only drops its target, so without an explicit close
the file is closed by whoever frees the handler last, with a ResourceWarning.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import uuid
from pathlib import Path

import pytest

from birdnet.acoustic.inference.file_writer import QueueFileWriter

pytestmark = pytest.mark.no_tf


def _writer_with_recorded_file_handler(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[QueueFileWriter, mp.queues.Queue, list[logging.FileHandler]]:
  opened: list[logging.FileHandler] = []

  class RecordingFileHandler(logging.FileHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
      super().__init__(*args, **kwargs)  # type: ignore[arg-type]
      opened.append(self)

  monkeypatch.setattr(logging, "FileHandler", RecordingFileHandler)
  ctx = mp.get_context("spawn")
  log_queue = ctx.Queue()
  stop_event = threading.Event()
  stop_event.set()
  writer = QueueFileWriter(
    session_id=uuid.uuid4().hex,
    log_queue=log_queue,
    logging_level=logging.DEBUG,
    log_file=tmp_path / "session.log",
    cancel_event=ctx.Event(),
    stop_event=stop_event,
  )

  return writer, log_queue, opened


def test_call_closes_the_log_file(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  writer, log_queue, opened = _writer_with_recorded_file_handler(tmp_path, monkeypatch)

  try:
    writer()
  finally:
    log_queue.close()
    log_queue.join_thread()

  assert len(opened) == 1
  # The list keeps the handler alive, so only an explicit close empties this.
  assert opened[0].stream is None, "the session log file was left open"


def test_a_failing_loop_still_detaches_and_closes(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  writer, log_queue, opened = _writer_with_recorded_file_handler(tmp_path, monkeypatch)

  def failing_loop(self: QueueFileWriter) -> None:
    raise RuntimeError("boom")

  monkeypatch.setattr(QueueFileWriter, "_run_main_loop", failing_loop)
  try:
    with pytest.raises(RuntimeError, match="boom"):
      writer()
  finally:
    log_queue.close()
    log_queue.join_thread()

  assert writer._logger.handlers == []
  assert opened[0].stream is None
