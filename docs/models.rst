Models
======

.. contents::
   :local:
   :depth: 2

Acoustic model V2.4 - June 2023
---------------

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
---------------------------------------

* updated species range model based on eBird data
* more accurate (spatial) species range prediction
* slightly increased long-tail distribution in the temporal resolution 
* see `this discussion post <https://github.com/birdnet-team/BirdNET-Analyzer/discussions/234>`__ for more details

Using older models
------------------

Older models are not supported in the current version of the package. If you need to use an older model, please refer to the `BirdNET-Analyzer repository <https://github.com/birdnet-team/BirdNET-Analyzer>`__.
