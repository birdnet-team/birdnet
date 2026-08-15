Models
======

.. contents::
   :local:
   :depth: 2

Acoustic model V2.4 - June 2023
-------------------------------

* more than 6,000 species worldwide
* covers frequencies from 0 Hz to 15 kHz with two-channel spectrogram (one for low and one for high frequencies)
* 0.826 GFLOPs, 50.5 MB as FP32
* enhanced and optimized metadata model
* global selection of species (birds and non-birds) with 6,522 classes (incl. 11 non-event classes)

Technical details
^^^^^^^^^^^^^^^^^

* 48 kHz sampling rate (we up- and downsample automatically and can deal with artifacts from lower sampling rates)
* we compute 2 mel spectrograms as input for the convolutional neural network:

    * first one has fmin = 0 Hz and fmax = 3000; nfft = 2048; hop size = 278; 96 mel bins
    * second one has fmin = 500 Hz and fmax = 15 kHz; nfft = 1024; hop size = 280; 96 mel bins

* both spectrograms have a final resolution of 96x511 pixels
* raw audio will be normalized between -1 and 1 before spectrogram conversion
* we use non-linear magnitude scaling as mentioned in `Schlüter 2018 <http://ceur-ws.org/Vol-2125/paper_181.pdf>`__
* V2.4 uses an EfficienNetB0-like backbone with a final embedding size of 1024
* See `this comment <https://github.com/birdnet-team/BirdNET-Analyzer/issues/177#issuecomment-1772538736>`__ for more details

Geo model (species range model) V2.4 - V2, Jan 2024
----------------------------------------------------

* updated species range model based on eBird data
* more accurate (spatial) species range prediction
* slightly increased long-tail distribution in the temporal resolution
* see `this discussion post <https://github.com/birdnet-team/BirdNET-Analyzer/discussions/234>`__ for more details

Acoustic model V3.0 (preview)
-----------------------------

.. note::

   The V3.0 acoustic model is currently a `*preview* <https://zenodo.org/records/20703646>`__ release (``preview3.1``) and may change before the final release.

* global selection of more than 11,000 classes (birds and non-birds)
* available in four backends, selectable via the ``backend`` argument of ``birdnet.load``:

    * ``tf`` - TFLite/LiteRT (CPU only), ``fp32`` and ``fp16``
    * ``pb`` - ProtoBuf (CPU/GPU), ``fp32``
    * ``pt`` - PyTorch (CPU/GPU), ``fp32``; requires ``pip install birdnet[pt]``
    * ``onnx`` - ONNX Runtime (CPU/GPU), ``fp32`` and ``fp16``; requires ``pip install birdnet[onnx]``

* supports both ``predict(..)`` and ``encode(..)`` on all backends
* multilingual common names in 30 languages

Technical details
^^^^^^^^^^^^^^^^^

* 32 kHz sampling rate (audio is automatically resampled)
* 3 s segments (96,000 samples) covering frequencies from 0 Hz to 15 kHz
* final embedding size of 1280

.. code-block:: python

  import birdnet

  # e.g. load the ONNX backend with FP16 precision
  model = birdnet.load("acoustic", "3.0", "onnx", precision="fp16")
  predictions = model.predict("example/soundscape.wav")

Geo model (species range model) V3.0
------------------------------------

* updated species range model (release v3.0.4) covering 14,082 classes - birds, insects, amphibians and mammals
* aligned with the V3.0 acoustic taxonomy
* available in four backends via the ``backend`` argument of ``birdnet.load``:

    * ``tf`` - TFLite (CPU only), ``int8``, ``fp16`` and ``fp32``
    * ``pb`` - ProtoBuf (CPU/GPU), ``fp32``
    * ``pt`` - PyTorch/TorchScript (CPU/GPU), ``fp32``; requires ``pip install birdnet[pt]``
    * ``onnx`` - ONNX Runtime (CPU/GPU), ``fp16`` and ``fp32``; requires ``pip install birdnet[onnx]``

The model's ``.tflite`` files contain select TensorFlow ops, so the ``tf`` backend
requires TensorFlow 2.18 or 2.19 and is not available via ``ai_edge_litert``
(``library="litert"``). On other TensorFlow versions use ``pb``, ``onnx`` or ``pt``.

.. code-block:: python

  import birdnet

  model = birdnet.load("geo", "3.0", "onnx")
  predictions = model.predict(42.5, -76.45, week=4)

Using older models
------------------

Older models are not supported in the current version of the package. If you need to use an older model, please refer to the `BirdNET-Analyzer repository <https://github.com/birdnet-team/BirdNET-Analyzer>`__.
