import importlib.metadata
from typing import Dict

version = importlib.metadata.version("birdnet")
print(version)

import absl.logging as absl_logging
from tensorflow import TensorSpec

absl_logging.set_verbosity(absl_logging.ERROR)  # absl-Backend
absl_logging.set_stderrthreshold("error")
Dict[str, TensorSpec(shape=(None, 6522), dtype=tf.float32, name="scores")]
