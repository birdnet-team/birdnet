import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import tflite_runtime.interpreter as tflite


class ModelBase():
  pass


# Frequency range. This is model specific and should not be changed.
SIG_FMIN: int = 0
SIG_FMAX: int = 15000


class ModelV2p4(ModelBase):
  def __init__(self, tflite_threads: int = 1) -> None:
    super().__init__()
    model_path = Path(
      "src/birdnet/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
    labels_path = Path("src/birdnet/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt")
    codes_file = Path("src/birdnet/checkpoints/eBird_taxonomy_codes_2021E.json")
    with codes_file.open(encoding="utf8") as f:
      self.codes = json.load(f)
    labels = labels_path.read_text("utf8").splitlines()
    self._labels = labels
    # [line.split(",")[1] for line in labels]

    # Load TFLite model and allocate tensors.
    self.interpreter = tflite.Interpreter(str(model_path.absolute()), num_threads=tflite_threads)
    # self.interpreter.allocate_tensors()

    # Get input tensor index
    input_details = self.interpreter.get_input_details()
    self.input_layer_index = input_details[0]["index"]

    # Get classification output
    output_details = self.interpreter.get_output_details()
    self.output_layer_index = output_details[0]["index"]

  @property
  def labels(self):
    return self._labels

  def predict(self, sample: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    assert sample.dtype == np.float32
    # Same method as embeddings

    self.interpreter.resize_tensor_input(self.input_layer_index, sample.shape)
    self.interpreter.allocate_tensors()

    # Make a prediction (Audio only for now)
    self.interpreter.set_tensor(self.input_layer_index, sample)
    self.interpreter.invoke()
    prediction = self.interpreter.get_tensor(self.output_layer_index)

    return prediction
