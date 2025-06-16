from birdnet_v2.base import ModelBase


class AcousticModelBase(ModelBase):
  def __init__(self, version: str, backend: str) -> None:
    super().__init__(version, backend, "acoustic")
