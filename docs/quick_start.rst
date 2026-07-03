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
  
Predict species for a given location and time
---------------------------------------------

.. code-block:: python
  
  import birdnet

  model = birdnet.load("geo", "2.4", "tf")

  predictions = model.predict(42.5, -76.45, week=4)

  predictions.to_txt("example/species.txt")

Predict species with a custom species list
------------------------------------------

.. code-block:: python
  
  import birdnet

  model = birdnet.load("acoustic", "2.4", "tf")

  predictions = model.predict(
      "example/soundscape.wav",
      custom_species_list="example/species.txt",
  )

  predictions.to_csv("example/predictions.csv")

Use a different model version or backend
----------------------------------------

The ``version`` and ``backend`` arguments of ``birdnet.load`` select the model. Besides V2.4, the V3.0 (preview) acoustic model is available in the ``tf``, ``pb``, ``pt`` and ``onnx`` backends, and the V3.0 geo model in the ``tf`` and ``pb`` backends. See :doc:`models` for the full support matrix.

.. code-block:: python

  import birdnet

  # V3.0 acoustic model via the ONNX backend (requires: pip install birdnet[onnx])
  model = birdnet.load("acoustic", "3.0", "onnx")

  predictions = model.predict("example/soundscape.wav")

  predictions.to_csv("example/predictions.csv")