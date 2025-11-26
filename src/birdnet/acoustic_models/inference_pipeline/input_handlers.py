from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from ordered_set import OrderedSet

from birdnet.acoustic_models.inference.producer import get_audio_duration_s

InputType = TypeVar("InputType")


class InputData(ABC):
  pass


class FileInputData(InputData):
  def __init__(self, file: Path) -> None:
    self.file = file

  def get_duration_s(self) -> float:
    return get_audio_duration_s(self.file)


class AudioInputData(InputData):
  def __init__(self, audio: np.ndarray, sample_rate: int) -> None:
    self.audio = audio
    self.sample_rate = sample_rate

  def get_duration_s(self) -> float:
    return self.audio.shape[0] / self.sample_rate


class InputDataHandler(Generic[InputType], ABC):
  @abstractmethod
  def push_to_analyzer_queue(self, queue: Queue) -> None:  # noqa: ANN401
    ...

  @abstractmethod
  def push_to_producer_queue(self, queue: Queue) -> None:  # noqa: ANN401
    ...

  @classmethod
  def iter_audio_durations(cls, analyzer_data: Any) -> Generator[float, None, None]: ...


class FileInputHandler(InputDataHandler):
  def __init__(self, files: OrderedSet[Path]) -> None:
    self._files = files

  def push_to_analyzer_queue(self, queue: Queue) -> None:  # noqa: ANN401
    queue.put(self._files)

  def push_to_producer_queue(self, queue: Queue) -> None:  # noqa: ANN401
    for input_idx, file_path in enumerate(self._files):
      queue.put((input_idx, file_path), block=False)

  @classmethod
  def iter_audio_durations(cls, analyzer_data: Any) -> Generator[float, None, None]:
    assert isinstance(analyzer_data, OrderedSet)
    for path in analyzer_data:
      assert isinstance(path, Path)
      audio_duration_s = get_audio_duration_s(path)
      yield audio_duration_s


class AudioInputHandler(InputDataHandler):
  def __init__(self, audio: list[np.ndarray], sample_rate: int) -> None:
    self._audio = audio
    self._sample_rate = sample_rate

  def push_to_analyzer_queue(self, queue: Queue) -> None:  # noqa: ANN401
    n_samples_list = [audio.shape[0] for audio in self._audio]
    queue.put((n_samples_list, self._sample_rate))

  def push_to_producer_queue(self, queue: Queue) -> None:  # noqa: ANN401
    for input_idx, audio in enumerate(self._audio):
      queue.put((input_idx, audio), block=False)

  @classmethod
  def iter_audio_durations(cls, analyzer_data: Any) -> Generator[float, None, None]:
    input_arrays, sample_rate = analyzer_data
    assert isinstance(sample_rate, int)
    assert isinstance(input_arrays, list)
    for input_array in input_arrays:
      assert isinstance(input_array, np.ndarray)
      audio_duration_s = input_array.shape[0] / sample_rate
      yield audio_duration_s
