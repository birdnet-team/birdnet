Quick start
===========

This guide will help you get started with BirdNET quickly. It covers installation, basic usage, and running benchmarks to evaluate performance.

.. contents::
   :local:
   :depth: 2

Installation
------------

To install BirdNET, you can use ``pip``. Open your terminal and run the following command:

.. code-block:: bash

   pip install birdnet --user
   
This command installs the BirdNET package along with its dependencies.

Predict species from audio file
-------------------------------

.. code-block:: python
  
  import birdnet

  model = birdnet.load("acoustic", "2.4", "tf")

  predictions = model.predict("example/soundscape.wav")

  predictions.to_csv("example/predictions.csv")

Predict species from multiple audio files in a directory
--------------------------------------------------------

.. code-block:: python
  
  import birdnet
  from birdnet import AudioDataset

  model = birdnet.load("acoustic", "2.4", "tf")

  predictions = model.predict("example/soundscapes/")

  predictions.to_csv("example/predictions.csv")

Limit prediction worker processes
---------------------------------

``predict`` starts one worker process per physical CPU core when ``n_workers`` is
omitted. Pass a fixed value when a scheduler or container limits the number of
processes available to the job:

.. code-block:: python

  import birdnet

  if __name__ == "__main__":
      model = birdnet.load("acoustic", "2.4", "tf")
      predictions = model.predict("example/soundscapes/", n_workers=2)
      predictions.to_csv("example/predictions.csv")

Each call to ``predict`` shuts down its producer and worker processes before it
returns, including when inference raises an exception. To process several batches
without restarting two workers for every batch, reuse a prediction session:

.. code-block:: python

  import birdnet

  if __name__ == "__main__":
      model = birdnet.load("acoustic", "2.4", "tf")

      with model.predict_session(n_workers=2) as session:
          first = session.run("example/soundscapes/day-1/")
          second = session.run("example/soundscapes/day-2/")

      first.to_csv("example/day-1.csv")
      second.to_csv("example/day-2.csv")
  
Predict species for a given location and time
---------------------------------------------

.. code-block:: python
  
  import birdnet

  model = birdnet.load("geo", "2.4", "tf")

  predictions = model.predict(42.5, -76.45, week=4)

  predictions.to_csv("example/location.csv")

Predict species with a custom species list
------------------------------------------

.. code-block:: python
  
  import birdnet

  model = birdnet.load("acoustic", "2.4", "tf")

  predictions = model.predict(
      "example/soundscape.wav",
      custom_species_list="example/species_list.txt",
  )

  predictions.to_csv("example/predictions.csv")

Show model download progress
----------------------------

Official models (plus their labels and, for V3.0, a shared taxonomy) are downloaded on first use inside ``birdnet.load(..)``. By default progress is only shown as a ``tqdm`` bar on stderr, which is invisible to a GUI application that redirects stderr to a log file. Register a process-wide callback to drive your own progress UI instead; while a callback is registered the tqdm bar is silenced. The callback receives a ``DownloadProgress`` snapshot:

* ``"started"`` once per attempt (a second ``"started"`` with a higher ``attempt`` is a retry -- reset your bar),
* ``"progress"`` while bytes arrive (throttled to ~10 per second),
* ``"retrying"`` before each back-off, with ``error`` and ``retry_in_s``,
* exactly one of ``"finished"`` / ``"failed"`` (with ``error``) at the end -- ``failed`` means the download gave up and ``load(..)`` raises right after.

.. code-block:: python

  import birdnet
  from birdnet import DownloadProgress

  def on_download_progress(p: DownloadProgress) -> None:
      # Runs on the thread that called birdnet.load(); hand off to your UI thread if needed.
      # A single load() may run several downloads (labels, taxonomy, model): key on p.description.
      if p.status == "started":
          ui.show_progress(p.description, attempt=p.attempt, of=p.max_attempts)
      elif p.status == "progress":
          ui.set_progress(p.fraction)  # None while the total size is unknown
      elif p.status == "retrying":
          ui.set_message(f"{p.error} - retrying in {p.retry_in_s:.0f} s")
      elif p.status == "finished":
          ui.hide_progress()
      elif p.status == "failed":
          ui.show_error(f"Could not download {p.description}: {p.error}")

  birdnet.set_download_progress_callback(on_download_progress)  # once, at application start

  model = birdnet.load("acoustic", "3.0", "onnx")  # downloads on first use

To cancel a running download from the UI, raise an exception inside the callback: the partial file is discarded, no retry is attempted, and your exception propagates out of ``load(..)``. ``set_download_progress_callback`` returns the previously registered callback; ``get_download_progress_callback`` reads it. Use ``birdnet.download_progress_callback(cb)`` as a ``with`` block instead if the callback should only apply to a specific piece of code:

.. code-block:: python

  import birdnet

  with birdnet.download_progress_callback(on_download_progress):
      model = birdnet.load("acoustic", "3.0", "onnx")

Use a different model version or backend
----------------------------------------

The ``version`` and ``backend`` arguments of ``birdnet.load`` select the model. Besides V2.4, the V3.0 (preview) acoustic model is available in the ``tf``, ``pb``, ``pt`` and ``onnx`` backends, and the V3.0 geo model in the ``tf``, ``pb`` and ``onnx`` backends. See :doc:`models` for the full support matrix.

.. code-block:: python

  import birdnet

  # V3.0 acoustic model via the ONNX backend (requires: pip install birdnet[onnx])
  model = birdnet.load("acoustic", "3.0", "onnx")

  predictions = model.predict("example/soundscape.wav")

  predictions.to_csv("example/predictions.csv")
