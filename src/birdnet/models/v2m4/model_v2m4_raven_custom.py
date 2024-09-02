import csv
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from ordered_set import OrderedSet

from birdnet.models.v2m4.model_v2m4_protobuf import (AudioModelV2M4ProtobufBase,
                                                     check_protobuf_model_files_exist,
                                                     get_custom_device,
                                                     try_get_gpu_otherwise_return_cpu)
from birdnet.types import Species
from birdnet.utils import sigmoid_inverse


class CustomRavenParser():
  def __init__(self, classifier_folder: Path, classifier_name: str) -> None:
    self._audio_model_path = classifier_folder / f"{classifier_name}"
    self._label_path = classifier_folder / f"{classifier_name}" / "labels" / "label_names.csv"

  @property
  def audio_model_path(self) -> Path:
    return self._audio_model_path

  @property
  def language_path(self) -> Path:
    return self._label_path

  def check_model_files_exist(self) -> bool:
    model_is_available = True
    model_is_available &= self._audio_model_path.is_dir()
    model_is_available &= self._label_path.is_file()
    model_is_available &= check_protobuf_model_files_exist(self.audio_model_path)
    return model_is_available


def get_species_from_raven_csv(path: Path) -> Generator[Species, None, None]:
  with path.open(newline='\n', encoding='utf-8', mode="r") as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      if len(row) != 2:
        raise ValueError(
          "Invalid input format detected! Expected species names in Raven model to be something like 'Card1,Cardinalis cardinalis_Northern Cardinal'.")
      code, description = row
      yield description


class CustomAudioModelV2M4Raven(AudioModelV2M4ProtobufBase):
  def __init__(self, classifier_folder: Path, classifier_name: str, /, *, custom_device: Optional[str] = None) -> None:
    parser = CustomRavenParser(classifier_folder, classifier_name)
    if not parser.check_model_files_exist():
      raise ValueError(
        f"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{classifier_folder.absolute()}' doesn't contain a valid raven classifier which has the name '{classifier_name}'!")

    device: tf.config.LogicalDevice
    if custom_device is None:
      device = try_get_gpu_otherwise_return_cpu()
    else:
      device = get_custom_device(custom_device)

    species_list = OrderedSet(get_species_from_raven_csv(parser.language_path))

    super().__init__(parser.audio_model_path, species_list, device)
    del parser

  def predict_species(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    prediction_np = super().predict_species(batch)
    # Raven models have an activation layer `keras.layers.Activation("sigmoid"))` which need to be reverted
    prediction_np = sigmoid_inverse(prediction_np)

    return prediction_np
