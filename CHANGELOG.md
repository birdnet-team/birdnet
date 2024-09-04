# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.6] - 2024-09-04

### Added

- Support for multiprocessing using `predict_species_within_audio_files_mp`

### Changed

- Separate `ModelV2M4TFLite` into `AudioModelV2M4TFLite` and `MetaModelV2M4TFLite`
- Separate `ModelV2M4Protobuf` into `AudioModelV2M4Protobuf` and `MetaModelV2M4Protobuf`
- Separate `ModelV2M4` into `AudioModelV2M4` and `MetaModelV2M4`
- Move v2.4 models to `birdnet.models.v2m4`
- Yield results of `predict_species_within_audio_file` instead of returning an OrderedDict
- Extracted method `predict_species_within_audio_file` and `predict_species_at_location_and_time` from their respective model
- set default value for `batch_size` to 100

## [0.1.5] - 2024-08-16

### Fixed

- Custom Raven audio model didn't return same results as custom TFLite model because of sigmoid layer
- TFLite meta model was not returning correct results

### Changed

- Rename `CustomModelV2M4TFLite` to `CustomAudioModelV2M4TFLite`
- Rename `CustomModelV2M4Raven` to `CustomAudioModelV2M4Raven`

## [0.1.4] - 2024-08-13

### Added

- Support to load custom TFLite models using `CustomModelV2M4TFLite`
- Support to load custom Raven (Protobuf) models using `CustomModelV2M4Raven`

## [0.1.3] - 2024-08-13

### Changed

- Make CUDA dependency optional, install with `birdnet[and-cuda]`

### Fixed

- Bugfix 'ERROR: Could not find a version that satisfies the requirement nvidia-cuda-nvcc-cu12 (Mac/Ubuntu/Windows)' (#4)

## [0.1.2] - 2024-08-07

### Added

- Add GPU support by introducing the Protobuf model (v2.4)

### Changed

- Rename class 'ModelV2M4' to 'ModelV2M4TFLite'
- 'ModelV2M4' defaults to Protobuf model now
- Sorting of prediction scores is now: score (desc) & name (asc)

### Fixed

- Bugfix output interval durations are now always of type 'float'

## [0.1.1] - 2024-08-02

### Added

- Add parameter 'chunk_overlap_s' to define overlapping between chunks (#3)

### Removed

- Remove parameter 'file_splitting_duration_s' instead load files in 3s chunks (#2)
- Remove 'librosa' dependency

## [0.1.0] - 2024-07-23

- Initial release

[Unreleased]: https://github.com/birdnet-team/birdnet/compare/v0.1.5...HEAD
[0.1.5]: https://github.com/birdnet-team/birdnet/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/birdnet-team/birdnet/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/birdnet-team/birdnet/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/birdnet-team/birdnet/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/birdnet-team/birdnet/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/birdnet-team/birdnet/releases/tag/v0.1.0
