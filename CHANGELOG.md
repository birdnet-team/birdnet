# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.11] - 2025-12-09

### Added

- Added model metadata to output
- Added more classes to `__init__.py` for easier imports
- Added skipping of unprocessable inputs

### Fixed

- Loading multiple GPUs in parallel processes was not possible
- Fixed hanging problem after error in processing occurred

### Changed

- Renamed many of the classes and functions for better clarity

## [0.2.10] - 2025-11-28

### Added

- Added support for Perch model v2

### Fixed

- Removed support for LiteRT on macOS ARM64 due to incompatibility issues

## [0.2.9] - 2025-11-27

### Added

- Added option to supervise progress using callback function during inference

## [0.2.8] - 2025-11-26

### Added

- Added option to predict and encode raw audio numpy arrays

## [0.2.7] - 2025-11-25

### Added

- Added parameter `speed` to control playback speed of audio during inference

## [0.2.5] - 2025-11-17 & [0.2.6] - 2025-11-21

### Bugfix

- Fixed issue with using ProtoBuf CPU backend and TensorFlow GPU being available
- Fixed #17: Issue on macOS with too long ring buffer names
- Fixed #19: `queue.qsize()` is not used anymore
- Fixed issue with hanging session because of logging
- Fixed issue with downloading same model simultaneously

### Changed

- Removed `litert` install option, now litert is always installed if possible
- Rename `tf` library to `tflite` to better reflect the usage of TFLite/LiteRT
- Improved prediction speed, esp. for half-precision models (+10 seg/s)
- Lowered dependencies
- Update `ai-edge-litert` to version 2.0.3 on `repro`
- Better download progress indication of model files
- Model loading in tests is done before running other tests

### Added

- Added `repro` option to be able to get reproducible results
- Added support for Python 3.13
- Added CI on GitHub Actions for testing on multiple OS and Python versions

### Removed

- Remove unused dependencies `numba` and `resampy`

## [0.2.4] - 2025-11-05

### Bugfixes

- Fixed issue with loading supported files from a folder

### Changed

- Increased half-precision prediction speed
- Set "pyarrow==22.0.0"
- Set "numpy==2.0.2" because of compatibility with `perch-hoplite`
- Set default "half_precision" parameter to False because of lower speed

### Added

- Add half precision to CLI

## [0.2.3] - 2025-11-04

### Added

- Added support for Python 3.12

### Changed

- Changed tensorflow to newest version 2.20.0

## [0.2.2] - 2025-11-03

### Added

- Added support for running multiple sessions in a row or in parallel using threading or multiprocessing
- Each session has its own logger and log file

### Changed

- Changed naming of the benchmark output files

## [0.2.1] - 2025-10-29

### Added

- Added parameter `is_raven` to load function to specify whether a custom Protobuf model is a Raven model or not
- Fix int8 acoustic model wrong inference parameters
- Added tests

## [0.2.0] - 2025-10-27

### Changed

- Refactored the whole codebase to be able to load model and predict scores in two separate steps

## [0.2.0a0] - 2025-07-29

### Changed

- Refactored the whole codebase

## [0.1.7] - 2025-03-19

### Changed

- Switched model download links from TUCcloud to Zenodo [#10](https://github.com/birdnet-team/birdnet/issues/10)

### Fixed

- Added check for mono files [#9](https://github.com/birdnet-team/birdnet/issues/9)

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

[Unreleased]: https://github.com/birdnet-team/birdnet/compare/v0.2.11...HEAD
[0.2.11]: https://github.com/birdnet-team/birdnet/compare/v0.2.10...v0.2.11
[0.2.10]: https://github.com/birdnet-team/birdnet/compare/v0.2.9...v0.2.10
[0.2.9]: https://github.com/birdnet-team/birdnet/compare/v0.2.8...v0.2.9
[0.2.8]: https://github.com/birdnet-team/birdnet/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/birdnet-team/birdnet/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/birdnet-team/birdnet/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/birdnet-team/birdnet/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/birdnet-team/birdnet/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/birdnet-team/birdnet/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/birdnet-team/birdnet/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/birdnet-team/birdnet/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/birdnet-team/birdnet/compare/v0.2.0a0...v0.2.0
[0.1.5]: https://github.com/birdnet-team/birdnet/compare/v0.1.5...v0.2.0a0
[0.1.5]: https://github.com/birdnet-team/birdnet/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/birdnet-team/birdnet/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/birdnet-team/birdnet/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/birdnet-team/birdnet/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/birdnet-team/birdnet/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/birdnet-team/birdnet/releases/tag/v0.1.0
