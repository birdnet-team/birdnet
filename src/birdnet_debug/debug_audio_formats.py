def read_audio(path: Path):
  # if path.suffix.lower() in (".wav", ".flac", ".ogg"):
  # return list(load_audio_in_chunks_with_overlap(path))
  return list(iter_audio_chunks_mono(path))


from __future__ import annotations

import ctypes
import os
from collections import deque
from collections.abc import Generator
from itertools import count
from multiprocessing import Queue, shared_memory
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event, Semaphore
from pathlib import Path
from typing import Iterator, Tuple, Union

import av  # pip install av
import av.audio.resampler
import numpy as np
import numpy.typing as npt
import soundfile as sf
from scipy.signal import resample

from birdnet.acoustic_models.inference.producer import (
  SF_FORMATS,
  get_chunks_with_overlap,
  resample_array,
)


def load_audio_in_chunks_with_overlap_v2(
  audio_path: Path,
  /,
  *,
  chunk_duration_s: float = 3,
  overlap_duration_s: float = 0,
  # read_duration_s: Optional[float] = None,
  target_sample_rate: int = 48000,
) -> Generator[npt.NDArray[np.float32], None, None]:
  assert audio_path.is_file()
  assert audio_path.suffix.upper() in SF_FORMATS

  sf_info = sf.info(audio_path)
  is_mono = sf_info.channels == 1
  assert is_mono

  sample_rate = sf_info.samplerate

  timestamps = get_chunks_with_overlap(
    float(sf_info.duration),
    float(chunk_duration_s),
    float(overlap_duration_s),
  )
  full_audio, _ = sf.read(audio_path, dtype=np.float32)
  full_audio = resample_array(full_audio, sample_rate, target_sample_rate)

  for start, end in timestamps:
    start_samples = round(start * target_sample_rate)
    end_samples = round(end * target_sample_rate)
    audio = full_audio[start_samples:end_samples]
    yield audio


def frame_to_mono_f32(frame: av.AudioFrame) -> np.ndarray:
  """
  Konvertiert ein PyAV-AudioFrame in ein **mono float32**-Array
  (values in [-1, 1]).  Unterstützt packed & planar, int & float.
  """
  layout = frame.layout.name  # z. B. 'mono', 'stereo', 'stereop'
  planar = layout.endswith("p")
  n_channels = frame.layout.channels

  # ---- Integer-Dekodierung möglichst beibehalten -----------------
  if "s16" in frame.format.name:  # 16-bit Integer
    pcm = frame.to_ndarray()  # shape (C, S) oder (S, C)
    # auf float32 normalisieren
    scale = 1 / 32768.0
    if planar:  # (C, S)
      mono = (pcm.astype(np.float32).mean(axis=0)) * scale
    else:  # (S, C)
      mono = (pcm.astype(np.float32).mean(axis=1)) * scale

  else:  # Float, 24-bit Int, …
    pcm = frame.to_ndarray(format="flt")  # packed float32 erzwingen
    mono = pcm.mean(axis=1)  # (S, C) → (S,)

  return mono


def iter_audio_chunks_mono(
  path: str | bytes,
  *,
  chunk_duration_s: float = 3.0,
  overlap_duration_s: float = 0.0,
) -> Generator[Tuple[int, np.ndarray, int], None, None]:
  """
  Stream an audio file in fixed-size mono chunks – **zero copy except at the
  exact chunk boundary**.

  Parameters
  ----------
  path
      File/URL which FFmpeg can decode.
  chunk_duration_s
      Chunk length in seconds (> 0).
  overlap_duration_s
      Overlap between consecutive chunks (0 ≤ overlap < chunk_duration_s).

  Yields
  ------
  (idx, mono_samples, sr)
      idx            – sequential chunk counter
      mono_samples   – ``float32`` NumPy array, shape (n_samples,)
      sr             – sample-rate of the file (no resampling)

  Notes
  -----
  * Only *complete* chunks are emitted – trailing partial data is ignored.
  * Down-mixing to mono is done **immediately** per decoded frame to avoid
    storing multi-channel data longer than necessary.
  * Works on Linux/macOS/Windows; any codec supported by FFmpeg.
  """
  if chunk_duration_s <= 0:
    raise ValueError("chunk_duration_s must be > 0")
  if not (0 <= overlap_duration_s < chunk_duration_s):
    raise ValueError("0 ≤ overlap < chunk_duration_s required")

  with av.open(path, "r") as container:
    stream = next(s for s in container.streams if s.type == "audio")
    sr: int = stream.rate
    chunk_samples = int(round(chunk_duration_s * sr))
    hop_samples = chunk_samples - int(round(overlap_duration_s * sr))

    buf: deque[np.ndarray] = deque()  # holds mono float32 frames
    buf_len = 0
    idx = 0

    for packet in container.demux(stream):
      for frame in packet.decode():
        if isinstance(frame, av.audio.frame.AudioFrame):
          layout = frame.layout
          channels = layout.channels
          if layout.nb_channels == 1:
            mono = frame.to_ndarray()[0]
            # ---- decode one frame, down-mix to mono early --------------
            # 's16' keeps decoding cheap; cast to float32 afterwards.
            # pcm = frame.to_ndarray()  # (samples, ch)
            # if pcm.ndim == 1:  # already mono
            #   mono = pcm.astype(np.float32) / 32768.0
            # else:
            #   # weighted average = simple mean here; shape (samples,)
            #   mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
            # mono = frame_to_mono_f32(frame)
            buf.append(mono)
            buf_len += mono.shape[0]

            # ---- spit out as many chunks as we can --------------------
            while buf_len >= chunk_samples:
              if len(buf) == 1:  # fast path
                chunk = buf[0][:chunk_samples]
                remainder = buf[0][hop_samples:]
              else:
                joined = np.concatenate(list(buf), axis=0)
                chunk = joined[:chunk_samples]
                remainder = joined[hop_samples:]

              yield idx, chunk, sr
              idx += 1

              buf.clear()
              if remainder.size:
                buf.append(remainder)
              buf_len = remainder.shape[0]
          else:
            raise ValueError()


def stream_audio(
  path: str,
  target_sr: int = 48_000,
  chunk_duration: float = 3.0,  # seconds
  dtype=np.float32,
) -> Iterator[np.ndarray]:
  """
  Yields (audio_chunk, sr) for *every* `chunk_duration` seconds in `path`.
  Works with any format FFmpeg can read, on every platform.
  """
  container = av.open(path, format=None, mode="r")
  stream = container.streams.audio[0]

  frames_per_chunk = int(chunk_duration * target_sr)
  chunk = []

  for packet in container.demux(stream):
    for frame in packet.decode():
      # frame = resampler.resample(frame)
      x = frame.to_ndarray()
      chunk.append(x)  # shape: (ch, n_samples)

      # Concatenate across frames until we hit the chunk size
      samples = np.concatenate(chunk, axis=1) if len(chunk) > 1 else chunk[0]

      while samples.shape[1] >= frames_per_chunk:
        out, samples = np.split(samples, [frames_per_chunk], axis=1)
        res = out.T.astype(dtype, copy=False)
        yield res

      chunk = [samples] if samples.size else []

  # tail (optional)
  if chunk:
    samples = np.concatenate(chunk, axis=1)
    if samples.size:
      yield samples.T.astype(dtype, copy=False)


if __name__ == "__main__":
  names = [
    "soundscape.wma",
    "soundscape.aac",
    "soundscape.m4a",
    "soundscape.wav",
    "soundscape_ulaw.wav",
    "soundscape_alaw.wav",
    "soundscape_24bit.wav",
    "soundscape.flac",
    "soundscape.ogg",
    "soundscape.opus",
    "soundscape.mp3",
    "soundscape.aiff",
    "soundscape.aifc",
    "soundscape.au",
  ]
  for name in names:
    path = Path(f"src/birdnet_v2_tests/audio_formats/{name}")
    print(f"Reading {path}...")
    audio_chunks = read_audio(path)
    print(f"Read {len(audio_chunks)} chunks from {path}.")
    if audio_chunks:
      print(f"First chunk: {audio_chunks[0]}...")
