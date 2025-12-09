from __future__ import annotations

from dataclasses import dataclass

from birdnet.acoustic_models.inference.benchmarking import (
  FullBenchmarkMetaBase,
  MinimalBenchmarkMetaBase,
)


@dataclass
class MinimalBenchmarkEmbMeta(MinimalBenchmarkMetaBase):
  pass


@dataclass
class FullBenchmarkEmbMeta(FullBenchmarkMetaBase):
  model_emb_dim: int
