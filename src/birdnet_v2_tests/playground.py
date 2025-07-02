import importlib.metadata
from typing import Dict

import soundfile as sf

print("\n".join(sf.available_formats().keys()))
