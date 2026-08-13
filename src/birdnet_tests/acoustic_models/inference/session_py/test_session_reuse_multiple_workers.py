"""Reusing a session must survive the shutdown wake-up hand-off across workers.

The last producer releases a single permit on `sem_filled` when it finishes, and
each worker passes one on as it exits. With more than one worker that wake-up
therefore travels a chain: worker N only leaves its `acquire(timeout=1.0)` once
worker N-1 has released. The relay leaves exactly one permit outstanding when the
last worker exits, which `RingBufferResources.reset()` drains at the start of the
next run -- carry it over instead and a worker acquires it, scans an empty ring
and aborts the run with `Analysis was cancelled`.

Every other session-reuse test in the suite runs `n_workers=1`, which never
exercises the relay. This runs several calls on one reused session with two
workers and a clip long enough to keep both of them claiming slots. Results only,
no timing -- the latency side is covered by `test_session_call_overhead.py`.
"""

import numpy as np

from birdnet.model_loader import load

SAMPLE_RATE = 48_000
# 30 s at the model's 3 s segment length: enough batches that both workers take
# slots, so the relay runs in a different order from one call to the next.
CLIP_DURATION_S = 30.0
EXPECTED_N_SEGMENTS = 10
N_WORKERS = 2
N_CALLS = 5
_MAX_ABS_DIFF = 1e-6


def _noise_clip() -> tuple[np.ndarray, int]:
  rng = np.random.default_rng(42)
  samples = rng.standard_normal(int(SAMPLE_RATE * CLIP_DURATION_S)).astype(np.float32)
  return samples * 0.1, SAMPLE_RATE


def test_reused_session_with_multiple_workers_keeps_returning_full_results() -> None:
  audio = _noise_clip()
  model = load("acoustic", "2.4", "tf", library="tflite")

  with model.predict_session(n_workers=N_WORKERS, top_k=None) as session:
    first = session.run_arrays(audio)
    assert first.species_probs.shape[1] == EXPECTED_N_SEGMENTS

    for call_nr in range(2, N_CALLS + 1):
      current = session.run_arrays(audio)
      # Not bit-exact: with two workers a segment can be inferred by either
      # interpreter instance, and two instances need not agree to the last bit
      # (XNNPACK sizes its thread pool from the visible cores, which changes
      # the reduction order). The invariant under test is that a reused session
      # keeps returning the same results, not that floats are identical.
      np.testing.assert_allclose(
        current.species_probs,
        first.species_probs,
        rtol=0,
        atol=_MAX_ABS_DIFF,
        err_msg=f"call {call_nr} of {N_CALLS} disagrees with the first call",
      )
