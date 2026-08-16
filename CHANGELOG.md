# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `birdnet.set_download_progress_callback(cb)` (and a scoped `birdnet.download_progress_callback(cb)` context manager) so embedding applications — e.g. a GUI with stderr diverted to a log file — can render their own progress for first-run model/label downloads instead of a tqdm bar nobody sees. Defaults to today's behavior when unregistered.

### Bugfixes

- Acoustic V3.0 confidences were sigmoid-squashed twice: the V3.0 exports already apply the sigmoid in-graph, so the pipeline's default sigmoid compressed every score into [0.5, 0.73]. `predict(..)` now returns the model's probabilities unchanged; `sigmoid_sensitivity` values other than 1.0 and `apply_softmax=True` raise a `ValueError` for V3.0, since both need logits the exports do not expose. A calibration test now pins an absolute confidence per backend.
- The acoustic V3.0 `pb` backend requested v2.4's SavedModel signatures, so every `predict(..)`/`encode(..)` died with `KeyError: 'basic'`. It now reads the V3.0 export's single `serving_default` signature, and new integration tests run the real SavedModel against the `onnx` backend.
- A worker killed while processing a batch — the realistic out-of-memory case — no longer wedges the surviving workers: a killed process leaves the shared ring-buffer lock permanently held, so waiters now notice a cancelled run and shut down cleanly, letting the liveness check report the death instead of hanging the call (#73).
- Teardown after a cancelled run no longer hangs on a message a killed child never finished writing: the parent drains the child-to-parent queues on background threads and joins its queue-reader threads with a bound (#77).
- A child killed while writing to the shared logging queue no longer keeps the surviving children from exiting: children on the cancellation path release the logging queue as their last act, at the cost of the few records still buffered.
- The progress display's slot and busy-worker gauges are now read without taking the counter's lock, so a killed process can no longer block the stats interval; a reading may be one count behind.
- Still open: the consumer's own read of the results queue can block on a message truncated by a killed worker (#83).

## [1.0.0] - 2026-08-14

### Added

- Added an `on_file_complete` callback to acoustic `predict(..)`/`encode(..)` and their session variants (all models: 2.4, 3.0, Perch V2), fired once per file as soon as it is fully processed with a single-file result — enabling streaming per-file persistence and live output. File inputs only; runs off the inference hot path, so throughput is unaffected (#57).
- Added the BirdNET V3.0 (preview) acoustic model in four backends — `tf`, `pb`, `pt` and `onnx` — all supporting `predict(..)` and `encode(..)`. Load via `birdnet.load("acoustic", "3.0", <backend>)`; `pt`/`onnx` require the new `birdnet[pt]`/`birdnet[onnx]` extras (#41).
- Added the BirdNET-Geomodel V3.0 (v3.0.4, 14,082 classes covering birds, insects, amphibians and mammals) in the `tf`, `pb`, `pt` and `onnx` backends. Load via `birdnet.load("geo", "3.0", <backend>)`; `pt`/`onnx` require the matching extras and run without TensorFlow. The `pt` backend applies the sigmoid the TorchScript export omits, so all four backends return the same probabilities (#41).
- Added an `apply_softmax` option to acoustic `predict(..)` (all models), mirroring `apply_sigmoid`: scores become a softmax over the model logits, useful for confidence scores (e.g. Perch V2). Defaults to `False` (#54).
- Added partial Python 3.14 support: as TensorFlow has no 3.14 wheels yet, `birdnet` installs without TensorFlow there and runs the TF-free backends — acoustic 3.0 and geo 3.0 via `onnx`/`pt`; TensorFlow-only paths raise a clear error instead of an `ImportError` (#55).

### Changed

- The inference pipeline now creates its processes with `spawn` by default on all platforms instead of inheriting Linux's `fork`, which could deadlock workers after TensorFlow had started its multi-threaded runtime. A start method the application fixed globally is honored, and `BIRDNET_START_METHOD` overrides both, so `fork`/`forkserver` remain available by explicit opt-in (#63).
- The V3.0 models now share the geomodel's versioned taxonomy (`taxonomy_v0.2-Jun2026.csv`), which resolves every geo label and matches the acoustic label file more closely than the previous pin. Estonian (`et`) was dropped from the V3.0 language list, as the new taxonomy has no Estonian column (#41).
- The progress callback now runs on a background thread with a copy of the caller's context (contextvars), matching the new `on_file_complete` callback (#53).

### Bugfixes

- A pipeline process that dies mid-run — typically killed by the operating system when memory runs out — is now reported with its name and exit code instead of leaving the call hanging forever. Not covered: a worker killed *while processing a batch* still deadlocks the surviving workers on Linux and macOS (#73).
- Fixed the progress callback's closing update: it reported zero processed segments for runs that had processed everything, and for a run without predictions published nothing at all, leaving the call waiting indefinitely (#75).
- Model, label and taxonomy downloads now retry with a growing back-off instead of failing on the first transient network fault; permanent client errors still fail immediately.
- Removed a fixed ~1 s barrier from every `run_arrays(..)` call — on a warm session a 3 s clip went from 1069 ms to 39 ms. Cancelled runs still tear down on the poll interval.
- Fixed an assertion firing in the prediction and encoding workers when the ring-buffer scan finds no readable slot, which aborted the run instead of taking the clean exit that was already there.
- Producers and the performance tracker no longer attach the ring buffers from inside a `fork` child, where `SharedMemory(create=False)` could block forever on a lock CPython does not reinitialize after `fork`.
- Fixed corrupt rows in acoustic prediction/encoding output caused by growing the internal result buffer with `numpy.ndarray.resize`, plus an off-by-one in the initial segment count (#50).
- Geo model v3.0 caches now self-heal across releases: a cached SavedModel or label files from an older release were not detected as stale, so a version bump could keep serving outdated labels (#41).
- V3.0 label files now record which taxonomy they were generated from and are regenerated when it changes. The taxonomy is shared, so the first model to fetch a new one made it look current for every other model, which kept serving the previous release's localized names (#41).

## [0.2.16] - 2026-05-09

### Added

- Add support for overriding BirdNET’s application-data directory via an environment variable `BIRDNET_APP_DATA`, enabling users to place downloaded models/benchmarks in a custom location (useful for deployments with restricted home directories or shared storage).

### Bugfixes

- Fixed acoustic inference session being aborted on macOS when stats were enabled: hardened parent/child memory tracking against `psutil.AccessDenied`, and replaced the two tracked semaphores with a wrapper that mirrors the count into shared memory so `get_value()` works on macOS (#39)
- Fixed float16 quantization of segment timestamps in prediction results, which caused up to ±0.05 s drift in CSV/DataFrame/Parquet output (#38, #42). Also closed an analogous hole in encoding results where a hop duration that is exactly representable in float16 (e.g. hop=1.5) could still produce drifting accumulated timestamps. Timestamps are now always materialized at >= float32 precision at the source.

## [0.2.15] - 2026-05-02

### Bugfixes

- Fix issue with float16 input durations and hop duration not being exactly representable, which caused rounding errors to accumulate across segments and thus wrong segment times in the output (#32)

## [0.2.14] - 2026-04-30

### Bugfixes

- Allow classifiers trained with hidden units and with append mode(#33, #22)

## [0.2.13] - 2026-04-06

### Changed

- Changed sigmoid function to match birdnet_analyzer by @Josef-Haupt

### Bugfixes

- Fixed issue #29

## [0.2.12] - 2026-02-22

### Added

- Added convenience functions to export embeddings by @Josef-Haupt
- Added some code documentation

### Changed

- Updated flat sigmoid to match birdnet_analyzer by @Josef-Haupt

### Bugfixes

- Fixed issue with one test on Python 3.11 & 3.12
- Fixed issue with building package in tox environments

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

[Unreleased]: https://github.com/birdnet-team/birdnet/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/birdnet-team/birdnet/compare/v0.2.16...v1.0.0
[0.2.16]: https://github.com/birdnet-team/birdnet/compare/v0.2.15...v0.2.16
[0.2.15]: https://github.com/birdnet-team/birdnet/compare/v0.2.14...v0.2.15
[0.2.14]: https://github.com/birdnet-team/birdnet/compare/v0.2.13...v0.2.14
[0.2.13]: https://github.com/birdnet-team/birdnet/compare/v0.2.12...v0.2.13
[0.2.12]: https://github.com/birdnet-team/birdnet/compare/v0.2.11...v0.2.12
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
[0.2.0a0]: https://github.com/birdnet-team/birdnet/compare/v0.1.7...v0.2.0a0
[0.1.7]: https://github.com/birdnet-team/birdnet/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/birdnet-team/birdnet/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/birdnet-team/birdnet/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/birdnet-team/birdnet/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/birdnet-team/birdnet/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/birdnet-team/birdnet/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/birdnet-team/birdnet/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/birdnet-team/birdnet/releases/tag/v0.1.0
