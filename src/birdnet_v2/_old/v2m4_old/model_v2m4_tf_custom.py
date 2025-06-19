from pathlib import Path
from typing import Optional

from birdnet.utils import get_species_from_file
from birdnet_v2.acoustic_models.v2m4_old.model_v2m4_tf import AcousticTFModelV2M4Base


class CustomTFParser:
  def __init__(self, classifier_folder: Path, classifier_name: str) -> None:
    self._audio_model_path = classifier_folder / f"{classifier_name}.tflite"
    self._label_path = classifier_folder / f"{classifier_name}_Labels.txt"

  @property
  def audio_model_path(self) -> Path:
    return self._audio_model_path

  @property
  def language_path(self) -> Path:
    return self._label_path

  def check_model_files_exist(self) -> bool:
    model_is_available = True
    model_is_available &= self._audio_model_path.is_file()
    model_is_available &= self._label_path.is_file()
    return model_is_available


class CustomAcousticTFModelV2M4(AcousticTFModelV2M4Base):
  def __init__(
    self,
    classifier_folder: Path,
    classifier_name: str,
    /,
    *,
    tflite_num_threads: Optional[int] = 1,
  ) -> None:
    parser = CustomTFParser(classifier_folder, classifier_name)
    if not parser.check_model_files_exist():
      raise ValueError(
        f"Values for 'classifier_folder' and/or 'classifier_name' are invalid! Folder '{classifier_folder.absolute()}' doesn't contain a valid TFLite classifier which has the name '{classifier_name}'!"
      )

    species_list = get_species_from_file(parser.language_path, encoding="utf8")

    super().__init__(parser.audio_model_path, species_list, tflite_num_threads)
