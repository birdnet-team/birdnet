from birdnet_v2.base import ModelBase


class GeoModelBase(ModelBase):
  def __init__(self, version: str) -> None:
    super().__init__(version)
