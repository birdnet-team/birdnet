from abc import ABC, abstractmethod
from typing import Literal, Self

MODEL_TYPE_ACOUSTIC = "acoustic"
MODEL_TYPE_GEO = "geo"

MODEL_VERSION_V2_4 = "2.4"
MODEL_BACKEND_TF = "tf"
MODEL_BACKEND_PB = "pb"

MODEL_TYPES = Literal["acoustic", "geo"]
MODEL_VERSIONS = Literal["2.4",]
MODEL_BACKENDS = Literal["tf", "pb"]


class ModelBase(ABC):
  def __init__(self) -> None:
    pass

  @classmethod
  @abstractmethod
  def get_version(cls) -> MODEL_VERSIONS: ...

  @classmethod
  @abstractmethod
  def get_backend(cls) -> MODEL_BACKENDS: ...

  @classmethod
  @abstractmethod
  def get_model_type(cls) -> MODEL_TYPES: ...
