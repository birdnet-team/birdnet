"""A TFLite interpreter stand-in exposing only what the layout resolver reads."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tensor:
  index: int
  name: str
  shape_signature: list[int]

  def details(self) -> dict:
    return {
      "index": self.index,
      "name": self.name,
      "shape": [1 if d < 0 else d for d in self.shape_signature],
      "shape_signature": list(self.shape_signature),
    }


@dataclass
class StubInterpreter:
  inputs: list[Tensor]
  outputs: list[Tensor]
  # Intermediate tensors, besides inputs and outputs.
  others: list[Tensor] = field(default_factory=list)
  # signature name -> {"inputs": [...], "outputs": [output name -> tensor]}
  signatures: dict[str, dict] = field(default_factory=dict)

  def get_input_details(self) -> list[dict]:
    return [t.details() for t in self.inputs]

  def get_output_details(self) -> list[dict]:
    return [t.details() for t in self.outputs]

  def get_tensor_details(self) -> list[dict]:
    return [t.details() for t in (*self.inputs, *self.others, *self.outputs)]

  def get_signature_list(self) -> dict:
    return {
      name: {"inputs": sig["inputs"], "outputs": list(sig["outputs"])}
      for name, sig in self.signatures.items()
    }

  def get_signature_runner(self, name: str) -> _StubRunner:
    return _StubRunner(self.signatures[name]["outputs"])


@dataclass
class _StubRunner:
  outputs: dict[str, Tensor]

  def get_output_details(self) -> dict[str, dict]:
    return {name: t.details() for name, t in self.outputs.items()}


def audio_input(samples: int) -> Tensor:
  return Tensor(0, "serving_default_x:0", [-1, samples])


def v3_0_export(n_species: int) -> StubInterpreter:
  """Layout of an analyzer export on the 3.0 base."""
  predictions = Tensor(1004, "StatefulPartitionedCall_1:1", [1, n_species])
  embeddings = Tensor(1002, "StatefulPartitionedCall_1:0", [1, 1280])
  return StubInterpreter(
    inputs=[audio_input(96_000)],
    outputs=[predictions, embeddings],
    signatures={
      "serving_default": {
        "inputs": ["x"],
        "outputs": {"embeddings": embeddings, "predictions": predictions},
      }
    },
  )


def v2_4_export(n_species: int, prediction_idx: int = 545) -> StubInterpreter:
  """Layout of an analyzer export on the 2.4 base: one logit output."""
  predictions = Tensor(prediction_idx, "StatefulPartitionedCall_1:0", [-1, n_species])
  return StubInterpreter(
    inputs=[Tensor(0, "serving_default_input_audio:0", [-1, 144_000])],
    outputs=[predictions],
    others=[Tensor(prediction_idx - 1, "model_1/GLOBAL_AVG_POOL/Mean", [-1, 1024])],
    signatures={
      "serving_default": {
        "inputs": ["input_audio"],
        "outputs": {"output_0": predictions},
      }
    },
  )
