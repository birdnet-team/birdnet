from abc import ABC, abstractmethod
from typing import Literal

MODEL_TYPE_ACOUSTIC = "acoustic"
MODEL_TYPE_GEO = "geo"
MODEL_TYPES = Literal["acoustic", "geo"]

MODEL_VERSION_V2_4 = "2.4"
MODEL_VERSIONS = Literal["2.4",]

MODEL_BACKEND_TF = "tf"
MODEL_BACKEND_PB = "pb"
MODEL_BACKENDS = Literal["tf", "pb"]

MODEL_PRECISION_INT8 = "int8"
MODEL_PRECISION_FLOAT16 = "fp16"
MODEL_PRECISION_FLOAT32 = "fp32"
MODEL_PRECISIONS = Literal["int8", "fp16", "fp32"]


class ModelBase(ABC):
  def __init__(self) -> None:
    super().__init__()

  @classmethod
  @abstractmethod
  def get_version(cls) -> MODEL_VERSIONS: ...

  @classmethod
  @abstractmethod
  def get_backend(cls) -> MODEL_BACKENDS: ...

  @classmethod
  @abstractmethod
  def get_model_type(cls) -> MODEL_TYPES: ...
