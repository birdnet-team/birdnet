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

Known limitations
----

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
