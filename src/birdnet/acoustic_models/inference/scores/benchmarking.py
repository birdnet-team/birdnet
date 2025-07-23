from __future__ import annotations

import inspect
import multiprocessing as mp
import platform
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import psutil

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)
from birdnet.backends import (
  litert_installed,
  tf_installed,
)
from birdnet.globals import (
  ACOUSTIC_MODEL_VERSIONS,
  MODEL_BACKENDS,
  MODEL_PRECISIONS,
  MODEL_TYPES,
)
from birdnet.local_data import get_package_version


@dataclass
class MinimalBenchmarkMeta(MinimalBenchmarkMetaBase):
  pass


@dataclass
class FullBenchmarkMeta(FullBenchmarkMetaBase):
  # Parameter
  param_top_k: int
  param_sigmoid_apply: bool
  param_sigmoid_sensitivity: float | None
  param_confidence_threshold_default: float | None
  param_confidence_threshold_custom: int
  param_custom_species: int
