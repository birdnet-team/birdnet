import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize(
  ("relative_path", "class_name"),
  [
    ("src/birdnet/acoustic/models/v2_4/model.py", "AcousticModelV2_4"),
    ("src/birdnet/acoustic/models/v3_0/model.py", "AcousticModelV3_0"),
  ],
)
def test_predict_documents_worker_limit_and_cleanup(
  relative_path: str, class_name: str
) -> None:
  module = ast.parse((REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8"))
  model = next(
    node
    for node in module.body
    if isinstance(node, ast.ClassDef) and node.name == class_name
  )
  predict = next(
    node
    for node in model.body
    if isinstance(node, ast.FunctionDef) and node.name == "predict"
  )
  docstring = ast.get_docstring(predict) or ""

  assert "n_workers" in docstring
  assert "physical CPU cores" in docstring
  assert "shuts down" in docstring
