General functionalities
========================

.. image:: _static/birdnet-structure.svg
   :alt: birdnet structure
   :align: center
   
The analysis pipeline processes **recordings** with five logically distinct components:

1. **Producers** – Read the recordings, split them into 3-second segments, group them to batches, and fill the buffer.
2. **Buffer** – An intermediate store that holds batches of 3-second audio segments.
3. **Workers** – Take batches from the buffer and perform inference with the model.
4. **Consumer** – Receives the probabilities calculated by the *Workers* and writes them to the result array.
5. **Result** – A three-dimensional matrix in which:

   - **Dimension 1** represents the recordings.
   - **Dimension 2** represents the consecutive 3-second segments.
   - **Dimension 3** represents the species covered by the model.
   - Each matrix cell stores the predicted probability for a given species in the corresponding segment of the file.

Parallelisation and Resource Management
---------------------------------------

* **Number of Processes** – The numbers of *Producer* and *Worker* processes are configurable. By default, one (1) *Producer* is launched, while the number of *Workers* equals the count of *physical* CPU cores in the system.

*Producers* and *Workers* run concurrently: *Producers* preload batches into the buffer, and *Workers* consume those batches for inference.
A *Producer* loads only as much audio as the buffer can hold, keeping RAM usage low because at any moment only the required 3-second segments are in memory.

* **Buffer Size** – By default, the buffer is set to twice the *Worker* count, ensuring that every *Worker* always has a pre-loaded batch to process and thus avoids idle time.
* **Model Backends** – Each worker loads its own instance of the inference model. On the CPU, both **TFLite** and **Protocol Buffers** (Protobuf) models can be used; Protobuf models can optionally run on the GPU.
* **Best Practice for CPU Inference** – For CPU-only execution on Linux, the number of *Worker* processes should not exceed the number of physical cores, as oversubscription typically leads to reduced performance. When running TFLite, keep the batch size to one (1); larger batches offer no throughput benefit.

Multiprocessing start method
----------------------------

The pipeline creates its processes with the ``spawn`` start method by default on
**all** platforms — it does *not* inherit Linux's platform default of ``fork``.
Forking a process after TensorFlow has started its multi-threaded runtime can
deadlock the child, so a plain ``model.predict(...)`` is safe out of the box
even when TensorFlow is already loaded.

The default can be overridden, in order of precedence:

1. Set the ``BIRDNET_START_METHOD`` environment variable to ``spawn``,
   ``forkserver`` or ``fork``.
2. Fix the method globally in your application **before** using birdnet, e.g.
   ``multiprocessing.set_start_method("fork")`` — an explicitly chosen method
   is always honored.

With ``fork`` (explicit opt-in), workers inherit a model loaded in the parent
process via copy-on-write, which avoids per-worker model loading — but you are
responsible for calling the pipeline before TensorFlow spawns threads, or
accepting the deadlock risk. With ``spawn`` or ``forkserver``, each worker
loads its model itself.

Because the default is ``spawn``, the standard Python rule for scripts applies
on every platform (it always did on macOS and Windows): entry-point code must
be guarded with ``if __name__ == "__main__":``.

TensorFlow startup output
-------------------------

TensorFlow prints a startup banner from native code — the
``absl::InitializeLog`` warning and the oneDNN notice — every time it is
imported. Every worker process imports it, so a single prediction emits dozens
of those lines. birdnet hides them by running the relevant imports with file
descriptor 2 redirected, which is the only level at which native output can be
intercepted: ``logging``, absl's verbosity and ``TF_CPP_MIN_LOG_LEVEL`` all act
above the descriptor and never see these writes.

If the import raises, the captured text is written to stderr, so a broken
TensorFlow installation still reports itself. Otherwise it is emitted on the
``birdnet`` logger at ``DEBUG``. No handler is attached by default, and the
worker processes that load the models have none at all, so that record is in
practice unavailable.

To see warnings that never raise — a CUDA library that could not be loaded,
say, which is why a GPU is silently not used — or to diagnose a crash during
model loading, re-run with ``BIRDNET_TF_VERBOSE=1``.

For a large unattended run, consider setting ``BIRDNET_TF_VERBOSE=1`` from the
start, so the scheduler's log keeps the record. The banner is emitted once per
worker process per session and does not grow with the amount of audio, so a
job over millions of files pays the same handful of lines as a job over one.

Set ``BIRDNET_TF_VERBOSE=1`` to switch the suppression off and get TensorFlow's
startup output unchanged; any value other than ``0`` or the empty string counts
as enabled. The other backends are unaffected, being quiet already.

Known limitations
-----------------

**End-time precision on the last segment of short files (≤ ~34 minutes).**
For memory efficiency, per-file durations are stored in the smallest float
dtype that covers their magnitude: ``float16`` for files up to 2\ :sup:`11` ≈
2048 s, ``float32`` for files up to 2\ :sup:`24` s (~194 days), ``float64``
beyond. The stored duration is used as the upper clamp when computing the
``end_time`` of the *last* segment of each file. Inside the float16 range
this rounding is visible: the largest representable float16 below ``X`` may
differ from ``X`` by up to one ULP — about 0.06 s near 128 s, 0.25 s near
1024 s, and 0.5 s near 2048 s. The error appears only on the very last
segment per file and only when the actual file duration is not exactly
representable in float16 (integer-second durations up to 2048 s are
exact). For files of one hour or longer the storage dtype is float32, where
the equivalent ULP is below 4 ms even at 12 h, so the effect is not
observable in practice.

All other timestamps (``start_time`` and ``end_time`` of every segment that
does not hit the clamp) are computed at ≥ float32 precision regardless of
file length.
