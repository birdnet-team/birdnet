import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import OrderedDict as ODType
from typing import Tuple

from birdnet.types import Species, SpeciesList


class ModelBase():
  pass


@dataclass()
class AnalysisResultBase():
  file_path: Path
  model_version: str
  language: str
  start_time: datetime.datetime
  end_time: datetime.datetime
  duration_seconds: float
  file_duration_seconds: float
  predictions: ODType[Tuple[float, float], ODType[Species, float]]
  available_species: SpeciesList
  filtered_species: Optional[SpeciesList]
