from importlib import import_module

import pytest

# Every name birdnet.utils.helper keeps importable, by the module defining it.
HOMES = {
  "birdnet.utils.downloads": [
    "ModelInfo",
    "check_protobuf_model_files_exist",
    "SOURCE_MARKER_NAME",
    "write_source_marker",
    "check_source_marker",
    "DownloadError",
    "download_file_tqdm",
    "sha256_file",
    "ensure_single_file_model",
  ],
  "birdnet.utils.filesystem": [
    "write_text_atomic",
    "directory_lock",
    "lock_heartbeat",
  ],
  "birdnet.utils.dtypes": [
    "UIntCType",
    "max_value_for_uint_dtype",
    "uint_ctype_from_dtype",
    "get_uint_dtype",
    "get_float_dtype",
    "upgrade_float_dtype_for_value",
    "get_lossless_float_dtype",
  ],
  "birdnet.utils.segments": [
    "apply_speed_to_duration",
    "apply_speed_to_samples",
    "get_hop_duration_s",
    "get_n_segments_speed",
    "duration_as_samples",
  ],
  "birdnet.utils.dsp": [
    "bandpass_signal",
    "fillup_with_silence",
    "flat_sigmoid_logaddexp_fast",
    "flat_logit_fast",
    "flat_softmax_fast",
  ],
  "birdnet.utils.audio_files": [
    "SF_FORMATS",
    "is_supported_audio_file",
    "get_supported_audio_files_recursive",
    "get_file_formats",
  ],
  "birdnet.utils.species_lists": [
    "get_species_from_file",
    "RAVEN_LABELS_NAME",
    "get_species_from_raven_labels",
    "validate_species_list",
  ],
  "birdnet.utils.csv_format": [
    "format_input_for_csv",
    "hms_centis_fast",
  ],
  "birdnet.core.backends": ["check_is_intel_macos"],
  "birdnet.core.base": ["get_hash"],
}

shim = import_module("birdnet.utils.helper")


@pytest.mark.no_tf
def test_all_lists_every_name() -> None:
  assert set(shim.__all__) == {name for names in HOMES.values() for name in names}


@pytest.mark.no_tf
def test_every_name_is_the_object_its_home_defines() -> None:
  for home, names in HOMES.items():
    module = import_module(home)
    for name in names:
      assert getattr(shim, name) is getattr(module, name), f"{home}.{name}"


@pytest.mark.no_tf
def test_exposes_nothing_beyond_all() -> None:
  public = {name for name in vars(shim) if not name.startswith("_")}
  assert public == set(shim.__all__)
