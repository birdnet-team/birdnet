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
