"""Every `_log` helper hands its arguments to `logging` unformatted.

At INFO the record is never created, so nothing is rendered; at DEBUG the
arguments are rendered into the record. A sentinel whose rendering is
observable tells the two apart.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from birdnet.acoustic.inference.core.consumer import Consumer
from birdnet.acoustic.inference.core.file_completion import FileCompletionDispatcher
from birdnet.acoustic.inference.core.input_analyzer import InputAnalyzer
from birdnet.acoustic.inference.core.perf_tracker import (
  PerformanceTracker,
  ProgressDispatcher,
)
from birdnet.acoustic.inference.core.producer import Producer
from birdnet.acoustic.inference.core.worker import WorkerBase

pytestmark = pytest.mark.no_tf

HELPERS = [
  WorkerBase,
  Producer,
  Consumer,
  FileCompletionDispatcher,
  InputAnalyzer,
  PerformanceTracker,
  ProgressDispatcher,
]


class Sentinel:
  def __init__(self) -> None:
    self.rendered = 0

  def __str__(self) -> str:
    self.rendered += 1
    return "rendered-sentinel"


@pytest.mark.parametrize("cls", HELPERS, ids=lambda c: c.__name__)
def test_log_does_not_render_its_arguments_at_info(
  cls: type, caplog: pytest.LogCaptureFixture
) -> None:
  logger = logging.getLogger(f"birdnet_tests.lazy_log.{cls.__name__}")
  caplog.set_level(logging.INFO, logger=logger.name)
  sentinel = Sentinel()

  cls._log(SimpleNamespace(_logger=logger, _pid=4242), "value %s", sentinel)

  assert sentinel.rendered == 0
  assert caplog.records == []


@pytest.mark.parametrize("cls", HELPERS, ids=lambda c: c.__name__)
def test_log_renders_its_arguments_at_debug(
  cls: type, caplog: pytest.LogCaptureFixture
) -> None:
  logger = logging.getLogger(f"birdnet_tests.lazy_log.{cls.__name__}")
  caplog.set_level(logging.DEBUG, logger=logger.name)
  sentinel = Sentinel()

  cls._log(SimpleNamespace(_logger=logger, _pid=4242), "value %s", sentinel)

  assert sentinel.rendered >= 1
  assert len(caplog.records) == 1
  assert caplog.records[0].getMessage().endswith(": value rendered-sentinel")
