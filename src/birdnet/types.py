from typing import OrderedDict as ODType
from typing import Tuple

Species = str
Language = str
Confidence = float
TimeInterval = Tuple[float, float]
SpeciesPrediction = ODType[Species, Confidence]
SpeciesPredictions = ODType[TimeInterval, SpeciesPrediction]
