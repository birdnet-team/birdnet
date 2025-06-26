# birdnet_batch_inference.py – raw‑audio version
from __future__ import annotations


# You'll need these imports in your own code

# Next two import lines for this demo only

import soundfile as sf  # pip install soundfile

# try:
#   import tflite_runtime.interpreter as tflite
# except ImportError:  # fallback to full TF (heavier)

from birdnet_v2.acoustic_models.v2_4.base import AcousticModelBaseV2_4


class AcousticPBModelV2_4(AcousticModelBaseV2_4):
  def __init__(self, lang_id: str) -> None:
    super().__init__()
