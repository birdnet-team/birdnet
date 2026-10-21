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

    * ``tf`` - TFLite (CPU only), ``fp32`` and ``fp16``; requires ``pip install birdnet[tf]``
    * ``pb`` - ProtoBuf (CPU/GPU), ``fp32``; requires ``pip install birdnet[tf]``
    * ``pt`` - PyTorch (CPU/GPU), ``fp32``; requires ``pip install birdnet[pt]``
    * ``onnx`` - ONNX Runtime (CPU/GPU), ``fp32`` and ``fp16``; part of the base install

* supports both ``predict(..)`` and ``encode(..)`` on all backends
* multilingual common names in 30 languages

Technical details
^^^^^^^^^^^^^^^^^

* 32 kHz sampling rate (audio is automatically resampled)
* 3 s segments (96,000 samples) covering frequencies from 0 Hz to 15 kHz
* final embedding size of 1280
* the ``tf`` and ``pb`` exports have a static batch dimension of 1, so
  ``batch_size`` above 1 raises a ``ValueError`` for them; ``onnx`` and ``pt``
  take any batch size
* the ``pb`` and ``onnx`` exports return NaN scores and embeddings for a segment
  with no variance (digital silence, a constant offset); ``tf`` and ``pt`` return
  finite values

.. code-block:: python

  import birdnet

  # e.g. load the ONNX backend with FP16 precision
  model = birdnet.load("acoustic", "3.0", "onnx", precision="fp16")
  predictions = model.predict("example/soundscape.wav")

Custom classifiers
------------------

Classifiers trained with `BirdNET-Analyzer <https://github.com/birdnet-team/BirdNET-Analyzer>`__
on the 2.4 or the 3.0 base load with ``birdnet.load_custom``. The analyzer exports
each classifier as a TFLite file (with a ``_Labels.txt`` next to it), as a Raven
SavedModel and as a detached head.

.. code-block:: python

  import birdnet

  # TFLite export; "auto" reads the base version from the file
  model = birdnet.load_custom(
    "acoustic", "auto", "tf", "CustomClassifier.tflite", "CustomClassifier_Labels.txt"
  )
  predictions = model.predict("example/soundscape.wav")
  embeddings = model.encode("example/soundscape.wav")

  # Raven export: SavedModel with the "basic" signature, sigmoid in the model, no embeddings
  raven = birdnet.load_custom(
    "acoustic",
    "3.0",
    "pb",
    "CustomClassifier",
    "CustomClassifier/labels/label_names.csv",
    is_raven=True,
  )

* ``birdnet.inspect_custom_tflite(path)`` describes a TFLite export without
  importing TensorFlow into the calling process: the base version (``"2.4"`` or
  ``"3.0"``, derived from the input length and the embeddings width), whether the
  classifier replaces the official species or appends to them (the labels file
  then lists the official species first), its number of species, whether it
  applies the sigmoid itself and whether it exposes embeddings.
* The tensors of a TFLite export are resolved from its ``serving_default``
  signature, so exports of every analyzer release load; the ``classifier_type``
  argument is accepted but no longer needed.
* 2.4 TFLite exports output logits and get the pipeline's sigmoid. 3.0 exports
  and Raven exports on either base apply the sigmoid inside their graph; their
  scores are returned as they are, and a ``sigmoid_sensitivity`` other than
  ``1.0`` is applied on logits recovered from those scores, so it has the same
  effect on every export (and ``apply_sigmoid=False`` ignores it on every
  export). ``apply_softmax=True`` raises a ``ValueError`` for them.
* Exports on the 3.0 base inherit the official SavedModel's static batch of 1
  (``batch_size`` above 1 raises). The Raven export also inherits its NaN scores
  on a segment without variance; the TFLite export returns finite values there.
* A Raven export's own ``labels/label_names.csv`` (rows of ``id,label``) is
  accepted as its species list; the id column is dropped.
* Detached heads (``*_detached.tflite``) take embeddings instead of audio and are
  rejected with a message naming them.

Geo model (species range model) V3.0
------------------------------------

* updated species range model (release v3.0.4) covering 14,082 classes - birds, insects, amphibians and mammals
* aligned with the V3.0 acoustic taxonomy
* available in four backends via the ``backend`` argument of ``birdnet.load``:

    * ``tf`` - TFLite (CPU only), ``int8``, ``fp16`` and ``fp32``; requires ``pip install birdnet[tf]``
    * ``pb`` - ProtoBuf (CPU/GPU), ``fp32``; requires ``pip install birdnet[tf]``
    * ``pt`` - PyTorch/TorchScript (CPU/GPU), ``fp32``; requires ``pip install birdnet[pt]``
    * ``onnx`` - ONNX Runtime (CPU/GPU), ``fp16`` and ``fp32``; part of the base install

The model's ``.tflite`` files contain select TensorFlow ops, so the ``tf`` backend
requires TensorFlow 2.18 or 2.19 and is not available via ``ai_edge_litert``
(``library="litert"``); the interpreters of TensorFlow 2.20 and 2.21 no longer
run select TensorFlow ops.
On other TensorFlow versions use ``pb``, ``onnx`` or ``pt``.

.. code-block:: python

  import birdnet

  model = birdnet.load("geo", "3.0", "onnx")
  predictions = model.predict(42.5, -76.45, week=4)

Using older models
------------------

Older models are not supported in the current version of the package. If you need to use an older model, please refer to the `BirdNET-Analyzer repository <https://github.com/birdnet-team/BirdNET-Analyzer>`__.
