# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added an `on_file_complete` callback to acoustic `predict(..)`, `predict_session(..)`, `encode(..)` and `encode_session(..)` (all models: 2.4, 3.0, Perch V2). It fires once per input file the moment that file is fully processed, receiving a single-file result (`AcousticFilePredictionResult` / `AcousticFileEncodingResult`); invalid files are reported with their input marked unprocessable. This enables streaming per-file persistence (e.g. resumable multi-file analysis) and live output. The callback runs on a background thread with a copy of the caller's context, off the inference hot path, so it does not regress throughput. File inputs only (not `run_arrays`); a callback that raises cancels the run.
- Added support for the BirdNET V3.0 (preview) acoustic model with four backends: TFLite/LiteRT (`tf`), ProtoBuf (`pb`), PyTorch (`pt`) and ONNX (`onnx`). Both `predict(..)` and `encode(..)` are supported on all backends. Load via `birdnet.load("acoustic", "3.0", <backend>)`. The `pt` and `onnx` backends require the new `birdnet[pt]` and `birdnet[onnx]` install extras (#41).
- Added support for the BirdNET-Geomodel V3.0 (release v3.0.3) with TFLite/LiteRT (`tf`, INT8/FP16/FP32), ProtoBuf (`pb`) and ONNX (`onnx`, FP16/FP32) backends. Load via `birdnet.load("geo", "3.0", <backend>)`. The `onnx` backend requires the `birdnet[onnx]` extra and also lets the geo model run without TensorFlow (e.g. on Python 3.14). A PyTorch backend is not available yet, as the released `.pt` is a training checkpoint rather than a TorchScript model (#41).
- Added an `apply_softmax` option to acoustic `predict(..)` (all models: 2.4, 3.0, Perch V2), mirroring `apply_sigmoid`. When enabled, output scores are the softmax over the model's logits, which is useful for obtaining confidence scores (e.g. for Perch V2). Defaults to `False` (#54).
- Added (partial) support for Python 3.14. TensorFlow does not yet ship Python 3.14 wheels, so on 3.14 `birdnet` installs without TensorFlow and supports the models that have a TensorFlow-free backend: the acoustic 3.0 model (`onnx`/`pt`) and the geo 3.0 model (`onnx`). The version cap `<3.14` was lifted and `tensorflow` is now only a dependency on Python ≤3.13. TensorFlow-only paths (`tf`/`pb` backends, the acoustic 2.4, geo 2.4 and Perch models) raise a clear, actionable error on 3.14 instead of an `ImportError`. Full support will follow once TensorFlow provides 3.14 wheels (#55).

### Changed

- The progress callback now runs on a background worker thread with a copy of the caller's context (contextvars) as captured when the call starts, matching the behavior of the new `on_file_complete` callback (#53).

### Bugfixes

- A pipeline process that dies mid-run is now reported instead of hanging the run forever. The parent waits for one sentinel per worker on the results queue and for each child's finish signal, both without a timeout, because a healthy run legitimately takes as long as the audio requires. A child killed from the outside (typically the OOM killer when memory runs out) or crashing in native code sends neither, so `predict(..)`/`encode(..)` blocked indefinitely with no output at all — indistinguishable from a run still in progress. The parent now checks child liveness once per second while waiting and fails the run with the process name and its exit code, pointing at `n_workers`/`batch_size` as the memory knobs.
- Model, label and taxonomy downloads now survive transient network faults. All official downloads go through a single helper that previously made exactly one attempt, so a connection reset, read timeout, truncated stream or server-side 5xx during the first `load()` of a model failed the call outright. The helper now retries up to five times with a growing back-off (5 s, 15 s, 30 s, 60 s) and still fails immediately on permanent client errors (4xx) like a wrong URL. The window is deliberately wide because the observed fault is GitHub's release endpoint refusing connections for tens of seconds rather than a single dropped packet; the wait is only paid on a download that is already failing. Interrupted attempts keep being written to a temp file and cleaned up, so a retry can never leave a corrupt model in the cache. Failed downloads now raise `DownloadError`, which subclasses `ValueError` and carries the HTTP `status_code`, so existing `except ValueError` handlers keep working unchanged.
- Removed a fixed ~1 s barrier from every `run_arrays(..)` call. Workers wait for work in `sem_filled.acquire(timeout=1.0)`, which a `multiprocessing.Event` cannot interrupt, so `all_producers_finished` was only observed once the poll interval elapsed and every run ended by sitting out that interval. The last producer now releases a permit when it finishes and each worker passes one on as it exits, so they wake immediately and take the `claimed_slot is None` exit that was already there. On a warm session a 3 s clip went from 1069 ms to 39 ms, 9 s from 1134 ms to 95 ms and 30 s from 1358 ms to 315 ms. This covers normal producer completion only; a cancelled run still tears down on the poll interval.
- Fixed `assert dur2_search_for_filled_slot is not None` firing in the prediction and encoding workers when the ring-buffer scan finds no readable slot. The duration was only assigned on the found-a-slot path, so the assertion ran before the `claimed_slot is None` branch below it could take the clean `all_producers_finished` exit, making that branch unreachable. The scan is now timed unconditionally.
- Producers and the performance tracker no longer attach the ring buffers from inside a `fork` child. `SharedMemory(create=False)` calls `multiprocessing.resource_tracker.register`, whose module-level lock CPython does not reinitialize after `fork`, so a child that inherits it held by a thread which does not exist in the child blocks on the attach forever. The workers already avoided this by attaching in the parent and inheriting the mappings (`WorkerBase.__init__`); producers and `PerformanceTracker` (started by `show_stats="progress"`/`"benchmark"`) now follow the same rule, which covers every remaining child-side attach in the pipeline. Measured on Linux: five attaches per producer child plus one per tracker child before, none after. This is a candidate cause of the intermittent 600 s hangs in the `fork` CI lane, where a wedged producer child logs its logging init and never reaches "Waiting for start signal" -- exactly the window the attaches sit in.
- Fixed corrupt rows in acoustic prediction and encoding output: growing the internal result buffer via `numpy.ndarray.resize` (together with an off-by-one in the initial segment count) could leave stale or uninitialized values in some segments. Buffers are now reallocated and copied, so all output rows are correct (#50).
- Geo model v3.0 caches now self-heal across releases: previously a cached ProtoBuf SavedModel or generated label files from an older release were never detected as stale, so a geo model version bump could keep serving outdated species labels/counts. The SavedModel now records its source release and the label files are validated against the current labels, so both are re-fetched/regenerated when they no longer match (#41).

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

[Unreleased]: https://github.com/birdnet-team/birdnet/compare/v0.2.16...HEAD
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
