"""Embedding Arrow tables carry the result settings as ordered schema metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet_tests.acoustic_models.inference.encoding.encoding_result_py.test_encoding_to_structured_array import (  # noqa: E501
  create_file_encoding_result,
)

pytestmark = pytest.mark.no_tf


def test_schema_metadata_keys_and_values_in_order() -> None:
  result = create_file_encoding_result(
    n_files=2, duration_s=9, segment_duration_s=3.0, overlap_duration_s=0.5
  )

  metadata = result.to_arrow_table().schema.metadata

  assert [(k.decode(), v.decode()) for k, v in metadata.items()] == [
    ("segment_duration_s", "3.0"),
    ("overlap_duration_s", "0.5"),
    ("speed", "1.0"),
    ("n_inputs", "2"),
    ("model_path", str(Path("/model/path"))),
    ("model_version", "v2.4"),
    ("model_fmin", "0"),
    ("model_fmax", "15000"),
    ("model_sr", "48000"),
    ("model_precision", "fp32"),
    ("embedding_dim", "6"),
  ]
