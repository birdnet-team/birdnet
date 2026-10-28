from pathlib import Path

from birdnet.utils.downloads import check_protobuf_model_files_exist


def test_check_protobuf_model_files_exist(tmp_path: Path) -> None:
  assert not check_protobuf_model_files_exist(tmp_path)

  variables = tmp_path / "variables"
  variables.mkdir()
  (tmp_path / "saved_model.pb").write_bytes(b"pb")
  (variables / "variables.data-00000-of-00001").write_bytes(b"data")
  (variables / "variables.index").write_bytes(b"index")

  assert check_protobuf_model_files_exist(tmp_path)


def test_check_protobuf_model_files_exist_missing_index(tmp_path: Path) -> None:
  variables = tmp_path / "variables"
  variables.mkdir()
  (tmp_path / "saved_model.pb").write_bytes(b"pb")
  (variables / "variables.data-00000-of-00001").write_bytes(b"data")

  assert not check_protobuf_model_files_exist(tmp_path)
