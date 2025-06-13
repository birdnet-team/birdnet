  from pathlib import Path
from typing import Literal
from birdnet_v2.models.v2m4.model_v2m4_tf import AcousticTFModelV2M4


class AcousticTFModel():
    version: str | None = None          # z. B. "v3"
    model_path: Path | None = None
    device: Literal["cpu", "gpu"] = "cpu"
    num_threads: int = 1

    # nicht-picklables werden im __post_init__ erzeugt
    _interp: tflite.Interpreter | None = field(init=False, default=None, repr=False)

    # -------- öffentliche API -----------------------------------
    def predict(self, batch: np.ndarray) -> np.ndarray:
      if self._interp is None:
          self._lazy_init()
      self._interp.set_tensor(self._in_idx, batch)
      self._interp.invoke()
      return self._interp.get_tensor(self._out_idx)

    # -------- Pickling-Magie ------------------------------------
    def __getstate__(self):
      state = self.__dict__.copy()
      state["_interp"] = None          # Interpreter NICHT mitnehmen
      return state

    def __setstate__(self, state):
      self.__dict__.update(state)      # _interp weiterhin None
      # Interpreter wird erst bei predict() erzeugt

    # -------- internal ------------------------------------------
    def _lazy_init(self):
      tflite_path = _resolve_model(self.version, self.model_path, self.device)
      self._interp = tflite.Interpreter(str(tflite_path),
                                        num_threads=self.num_threads)
      self._interp.allocate_tensors()
      self._in_idx  = self._interp.get_input_details()[0]["index"]
      self._out_idx = self._interp.get_output_details()[0]["index"]

def load(spec: str, *, model_path: str | None = None):
  if spec == "acoustic/v2.4+tf@cpu":
    result = AcousticTFModelV2M4()
  elif spec == "geo/v3+pb@gpu":
    pass
  