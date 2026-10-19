"""Type-check calls against `model_loader.pyi`: every combination the loader
accepts must type-check and infer its concrete model class, and combinations it
rejects must fail."""

import os
import re
from pathlib import Path

import pytest
from mypy import api as mypy_api

import birdnet

pytestmark = pytest.mark.no_tf

STUB = Path(birdnet.__file__).with_name("model_loader.pyi")

ALL_CLASSIFIER_TYPES = ("replace", "append", "replace_hidden", "append_hidden")
BOTH_LIBRARIES = ("tflite", "litert")

# The accepted matrix as implemented by the `_load_*` branches of
# birdnet/model_loader.py: (model_type, version, backend, precisions, accepted
# keyword values, class(es) the call must infer).
LOAD_ACCEPTED: list[tuple[str, str, str, tuple[str, ...], dict, tuple[str, ...]]] = [
  ("acoustic", "2.4", "tf", ("int8", "fp16", "fp32"),
   {"library": BOTH_LIBRARIES, "lang": ("en_uk",)}, ("AcousticModelV2_4",)),
  ("acoustic", "2.4", "pb", ("fp32",), {"lang": ("en_uk",)}, ("AcousticModelV2_4",)),
  ("acoustic", "3.0", "tf", ("fp32", "fp16"),
   {"library": BOTH_LIBRARIES, "lang": ("es_mx",)}, ("AcousticModelV3_0",)),
  ("acoustic", "3.0", "pb", ("fp32",), {"lang": ("es_mx",)}, ("AcousticModelV3_0",)),
  ("acoustic", "3.0", "pt", ("fp32",), {"lang": ("es_mx",)}, ("AcousticModelV3_0",)),
  ("acoustic", "3.0", "onnx", ("fp32", "fp16"), {"lang": ("es_mx",)},
   ("AcousticModelV3_0",)),
  ("geo", "2.4", "tf", ("fp32",), {"library": BOTH_LIBRARIES, "lang": ("en_uk",)},
   ("GeoModelV2_4",)),
  ("geo", "2.4", "pb", ("fp32",), {"lang": ("en_uk",)}, ("GeoModelV2_4",)),
  # check_tf_library_for_v3_0 rejects "litert"
  ("geo", "3.0", "tf", ("int8", "fp16", "fp32"),
   {"library": ("tflite",), "lang": ("es_mx",)}, ("GeoModelV3_0",)),
  ("geo", "3.0", "pb", ("fp32",), {"lang": ("es_mx",)}, ("GeoModelV3_0",)),
  ("geo", "3.0", "pt", ("fp32",), {"lang": ("es_mx",)}, ("GeoModelV3_0",)),
  ("geo", "3.0", "onnx", ("fp32", "fp16"), {"lang": ("es_mx",)}, ("GeoModelV3_0",)),
]  # fmt: skip

# Same for the `_load_custom_*` branches.
LOAD_CUSTOM_ACCEPTED: list[
  tuple[str, str, str, tuple[str, ...], dict, tuple[str, ...]]
] = [
  ("acoustic", "2.4", "tf", ("int8", "fp16", "fp32"),
   {"library": BOTH_LIBRARIES, "classifier_type": ALL_CLASSIFIER_TYPES},
   ("AcousticModelV2_4",)),
  ("acoustic", "2.4", "pb", ("fp32",), {"is_raven": (True, False)},
   ("AcousticModelV2_4",)),
  ("acoustic", "auto", "tf", ("int8", "fp16", "fp32"),
   {"library": BOTH_LIBRARIES, "classifier_type": ALL_CLASSIFIER_TYPES},
   ("AcousticModelV2_4", "AcousticModelV3_0")),
  ("acoustic", "3.0", "tf", ("fp32", "fp16"),
   {"library": BOTH_LIBRARIES, "classifier_type": ALL_CLASSIFIER_TYPES},
   ("AcousticModelV3_0",)),
  ("acoustic", "3.0", "pb", ("fp32",), {"is_raven": (True, False)},
   ("AcousticModelV3_0",)),
  ("acoustic", "3.0", "pt", ("fp32",), {}, ("AcousticModelV3_0",)),
  ("acoustic", "3.0", "onnx", ("fp32", "fp16"), {}, ("AcousticModelV3_0",)),
  ("geo", "2.4", "tf", ("fp32",), {"library": BOTH_LIBRARIES}, ("GeoModelV2_4",)),
  ("geo", "2.4", "pb", ("fp32",), {}, ("GeoModelV2_4",)),
  ("geo", "3.0", "tf", ("int8", "fp16", "fp32"), {"library": ("tflite",)},
   ("GeoModelV3_0",)),
  ("geo", "3.0", "pb", ("fp32",), {}, ("GeoModelV3_0",)),
  ("geo", "3.0", "pt", ("fp32",), {}, ("GeoModelV3_0",)),
  ("geo", "3.0", "onnx", ("fp32", "fp16"), {}, ("GeoModelV3_0",)),
]  # fmt: skip

OTHER_ACCEPTED: list[tuple[str, tuple[str, ...]]] = [
  ('load_perch_v2("GPU")', ("AcousticModelPerchV2",)),
  ('inspect_custom_tflite("m.tflite", library="litert")', ("CustomTFLiteInfo",)),
  ('load("acoustic", "3.0", backend)', ("AcousticModelV3_0",)),
  ('load("geo", "3.0", backend)', ("GeoModelV3_0",)),
  ('load_custom("acoustic", "3.0", backend, "m", "s")', ("AcousticModelV3_0",)),
]

# Each is rejected at runtime by model_loader.py.
REJECTED = [
  'load("acoustic", "2.4", "pb", precision="fp16")',
  'load("acoustic", "2.4", "pt")',
  'load("acoustic", "3.0", "tf", precision="int8")',
  'load("acoustic", "3.0", "pt", precision="fp16")',
  'load("acoustic", "3.0", "pb", library="tflite")',
  'load("acoustic", "3.0", "onnx", lang="en_uk")',
  'load("geo", "2.4", "tf", precision="int8")',
  'load("geo", "3.0", "tf", library="litert")',
  'load("geo", "3.0", "pb", precision="int8")',
  'load("geo", "3.0", "pt", precision="fp16")',
  'load_custom("acoustic", "2.4", "tf", "m", "s", is_raven=True)',
  'load_custom("acoustic", "3.0", "pt", "m", "s", precision="fp16")',
  'load_custom("acoustic", "3.0", "onnx", "m", "s", library="tflite")',
  'load_custom("acoustic", "auto", "onnx", "m", "s")',
  'load_custom("geo", "2.4", "pb", "m", "s", precision="fp16")',
  'load_custom("geo", "3.0", "tf", "m", "s", library="litert")',
  'load_custom("geo", "3.0", "pb", "m", "s", precision="int8")',
  'load_perch_v2("TPU")',
  'inspect_custom_tflite("m.tflite", library="tf")',
]

_MESSAGE = re.compile(r"^(.*?):(\d+): (error|note): (.*)$")
_REVEALED = re.compile(r'Revealed type is "(.*)"')


def _accepted_calls() -> list[tuple[str, tuple[str, ...]]]:
  calls = list(OTHER_ACCEPTED)
  for func, extra_args, matrix in (
    ("load", "", LOAD_ACCEPTED),
    ("load_custom", ', "m", "s"', LOAD_CUSTOM_ACCEPTED),
  ):
    for model_type, version, backend, precisions, kwargs, expected in matrix:
      head = f'{func}("{model_type}", "{version}", "{backend}"{extra_args}'
      if func == "load_custom":
        kwargs = {**kwargs, "check_validity": (False,)}
      arguments = [""]
      arguments += [f', precision="{precision}"' for precision in precisions]
      arguments += [
        f", {name}={value!r}" for name, values in kwargs.items() for value in values
      ]
      calls += [(f"{head}{argument})", expected) for argument in arguments]
  return calls


def test_stub_matches_loader(tmp_path: Path) -> None:
  accepted = _accepted_calls()
  lines = [
    "from birdnet import inspect_custom_tflite, load, load_custom, load_perch_v2",
    "from birdnet import MODEL_BACKENDS",
    # a backend that is not a literal resolves through the combined overloads
    'backend: MODEL_BACKENDS = "tf"',
  ]
  expected_by_line: dict[int, tuple[str, ...]] = {}
  for call, expected in accepted:
    lines.append(f"reveal_type({call})")
    expected_by_line[len(lines)] = expected
  rejected_lines: dict[int, str] = {}
  for call in REJECTED:
    lines.append(call)
    rejected_lines[len(lines)] = call
  module = tmp_path / "stub_calls.py"
  module.write_text("\n".join(lines) + "\n", encoding="utf-8")
  config = tmp_path / "mypy.ini"
  config.write_text("[mypy]\n", encoding="utf-8")

  stdout, stderr, _ = mypy_api.run(
    [
      str(module),
      str(STUB),
      "--config-file",
      str(config),
      "--cache-dir",
      os.devnull,
      "--follow-imports=silent",
      "--no-error-summary",
      "--no-pretty",
    ]
  )
  assert stderr == ""

  errors: dict[int, list[str]] = {}
  revealed: dict[int, str] = {}
  stub_errors: list[str] = []
  for out_line in stdout.splitlines():
    match = _MESSAGE.search(out_line)
    if match is None:
      continue
    file, lineno, kind, message = match[1], int(match[2]), match[3], match[4]
    if not file.endswith("stub_calls.py"):
      if kind == "error":
        stub_errors.append(out_line)
      continue
    if kind == "error":
      errors.setdefault(lineno, []).append(message)
    elif message.startswith("Revealed type is"):
      revealed[lineno] = message

  unexpected = {
    lines[n - 1]: msgs for n, msgs in errors.items() if n not in rejected_lines
  }
  assert unexpected == {}
  for lineno, call in rejected_lines.items():
    assert len(errors.get(lineno, [])) == 1, f"{call} must be rejected once"
  # The stub itself must type-check: its defaults have to fit the Literal params.
  assert stub_errors == []
  for lineno, expected in expected_by_line.items():
    message = revealed.get(lineno, "")
    assert "Any" not in message, lines[lineno - 1]
    found = _REVEALED.search(message)
    assert found is not None, (lines[lineno - 1], message)
    names = sorted(part.strip().rsplit(".", 1)[-1] for part in found[1].split(" | "))
    assert names == sorted(expected), (lines[lineno - 1], message)
