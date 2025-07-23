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
class MinimalBenchmarkEmbMeta(MinimalBenchmarkMetaBase):
  pass


@dataclass
class FullBenchmarkEmbMeta(FullBenchmarkMetaBase):
  pass
