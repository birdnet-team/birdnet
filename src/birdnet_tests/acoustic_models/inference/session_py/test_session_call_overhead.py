"""A warm session must not add a fixed per-call barrier to `run_arrays(..)`.

Workers park in `sem_filled.acquire(timeout=1.0)` and cannot be woken by an
event, so before the last producer released a wake-up permit on completion --
relayed onwards by each worker as it exits -- every `run_arrays(..)` call sat
out a full poll interval no matter how small the payload: ~1.05 s for a 3 s clip
of which ~40 ms was inference.

This guards the regression by wall clock because the defect is wall clock. It
asserts on the fastest of several calls rather than the median: the defect was a
hard floor under every call, so the minimum tests the invariant directly and a
single scheduling hiccup on a contended CI runner cannot fail it. The threshold
is deliberately far from both sides: the old floor was a hard ~1.0 s, while a
3 s clip on a slow CI worker infers in well under 0.5 s.
"""

import time

import numpy as np

from birdnet.model_loader import load

SAMPLE_RATE = 48_000
SEGMENT_DURATION_S = 3.0
# The worker's poll interval is 1.0 s; anything at or above it means the barrier
# is back. Halved to leave room for a loaded CI worker without going near it.
MAX_FASTEST_CALL_DURATION_S = 0.5
N_CALLS = 5


def _noise_clip(seconds: float) -> np.ndarray:
  rng = np.random.default_rng(42)
  return rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32) * 0.1


def test_warm_session_run_arrays_has_no_fixed_second_long_barrier() -> None:
  audio = _noise_clip(SEGMENT_DURATION_S)
  model = load("acoustic", "2.4", "tf", library="tflite")

  with model.predict_session(n_workers=1, top_k=None) as session:
    session.run_arrays((audio, SAMPLE_RATE))  # warm up the interpreter

    durations = []
    for _ in range(N_CALLS):
      start = time.perf_counter()
      session.run_arrays((audio, SAMPLE_RATE))
      durations.append(time.perf_counter() - start)

  fastest = min(durations)
  assert fastest < MAX_FASTEST_CALL_DURATION_S, (
    f"fastest run_arrays duration {fastest:.3f} s over {N_CALLS} calls on a warm "
    f"session; a fixed per-call barrier has been reintroduced. "
    f"Samples: {[round(d, 3) for d in durations]}"
  )
