# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project

Python library (`src` layout) for identifying bird species by their sounds (BirdNET). Python 3.11–3.14 (3.14 is the TensorFlow-free surface only: onnx/pt backends, the `tf` backend on ai-edge-litert via `library="litert"`, and friendly errors on the remaining TF paths).

- `src/birdnet` — the library
- `src/birdnet_tests` — pytest suite (not shipped)
- `src/birdnet_benchmark` — benchmark CLI (`birdnet-benchmark` entry point)
- `benchmarks/` — self-contained cross-version benchmark harness (not shipped); run it per version in that version's venv to compare releases, see `benchmarks/README.md`

## Environment setup

- System dependency: libsndfile (`apt-get install libsndfile1` / `brew install libsndfile` / `choco install libsndfile`).
- Install: `uv pip install -e '.[tests,tf]'` (or plain pip). TensorFlow is optional (`tf` extra; `and-cuda` implies it) and most of the suite needs it — without it only the `no_tf`/`litert` tests run, the rest is skipped by `conftest.py`. Other extras: `pt` (torch); `onnx` is a no-op alias (onnxruntime is a base dependency); the `repro` extra pins exact versions and conflicts with normal dev. On Python 3.14 add `ai-edge-litert` by hand (tox does) or the litert `no_tf` tests skip.
- Official models auto-download on first load (~3 GB for the full set). The cache location is controlled by the `BIRDNET_APP_DATA` env var — set it to a persistent path in ephemeral environments. `pytest -m "not repro and load_model" -n auto` prefetches everything the tests need (in a TensorFlow-free env: `-m "load_model and (no_tf or litert)"`).

## Commands

```sh
# lint / format / type check (2-space indent, line length 88; ruff requires type annotations)
ruff check src/birdnet
ruff format src/birdnet
mypy   # configured via pyproject to check the birdnet package

# fast local test run (skips model downloads, litert, gpu, repro, fork)
pytest -m "not repro and not load_model and not litert and not gpu and not fork" -n auto

# single test
pytest src/birdnet_tests/path/to/test_file.py::test_name

# full matrix (py311-314 + py312-repro + py313-notf), used by CI
tox
```

### Test markers (ordering matters)

- `load_model` — downloads all models; run first before other tests.
- `litert` — must run in a separate pytest process: `ai_edge_litert` cannot be imported after TensorFlow (`pytest -m litert -n auto`).
- `gpu` — run sequentially (`-n 1`).
- `repro` — requires the exact pinned versions from the `repro` extra (Python 3.12, CPU only, not macOS Intel).
- `fork` — forces the fork start method; must run serially and in-process (`-n 0`), never in the parallel phase: forking after TensorFlow is loaded can wedge or segfault the child (see `conftest.py`). Fork support is best-effort — a hung fork test on macOS is likely the known TF limitation, not your change.
- `no_tf` — the TensorFlow-free surface; the only tests that run on Python 3.14, and (with `litert`) in the TF-free `py313-notf` lane. Anything not marked `no_tf` or `litert` is skipped when TensorFlow is not installed.
- `tf` — a `litert`-marked test that needs TensorFlow after all (e.g. reaches the `pb` guard); skipped in the TF-free lanes.

Per-test timeout is 600 s (thread method, kills the process on hang); worker restarts are disabled (`--max-worker-restart=0`).

## Conventions

- User-facing fixes and features get a `CHANGELOG.md` entry under `[Unreleased]` (Keep a Changelog format): one or two sentences covering cause and effect, not just "fixed X". Only breaking changes may run to a paragraph, and even those are covered more fully in the release notes; deep mechanics belong in the commit message. Match `[1.0.0]` for length. Cite the PR, and any issue it closes, as markdown links (`[#99](https://github.com/birdnet-team/birdnet/pull/99)`) — GitHub does not autolink a bare `#99` in a repository file.
- Code comments: short and current-state only — a constraint, a non-obvious why, or a measured value that justifies a bound. No history ("once was", "used to fail") and no narration; that belongs in commit messages and the changelog. A comment that adds nothing beyond the line it annotates is deleted, not kept — this applies to config files (workflow YAML, `pyproject.toml`, tox) as much as to Python.
- Tests mirror the source layout: `<module>_py/` directories, one file per method/behavior, optionally grouped in a `ClassName/` directory (e.g. `inference_pipeline/resources_py/RingBufferResources/test_reset.py`).
- Timing-sensitive tests assert invariants, not distributions — e.g. guard a latency floor with `min(durations)`, not a mean/median, so a loaded CI runner cannot flake it.

## Architecture

- Two model domains with parallel structure: `acoustic/` (species classification + embeddings from audio) and `geo/` (species presence from lat/lon/week). Each has `models/` (per version: `v2_4`, `v3_0`, plus acoustic-only `perch_v2`) and `inference/`.
- Public API is exported from `birdnet/__init__.py`. Entry points are `birdnet.load(model_type, version, backend)`, `load_custom`, and `load_perch_v2` in `model_loader.py` — keep the `model_loader.pyi` stub in sync when changing signatures.
- `core/backends.py` holds the backend abstraction: `TFBackend` (TFLite/LiteRT), `PBBackend` (ProtoBuf SavedModel), `TorchBackend`, `OnnxBackend`, plus `BackendLoader` and TF/torch/onnx device + import helpers. Torch and ONNX are optional extras (`pt`, `onnx`); LiteRT availability is platform-dependent.
- Acoustic inference runs through session objects (`AcousticPredictionSession`, `AcousticEncodingSession`) driven by prediction/encoding strategies and multiprocessing (`acoustic/inference/process_manager.py`). Result objects export to CSV/Parquet/Arrow/etc.
- Official models auto-download on first load (what `load_model` tests exercise).
- CI keys its ~3 GB model cache on the download set — every quoted `https://` URL and every `dl_size`/`*_DL_SIZE*`/`*_DOWNLOAD_SIZE*` and `sha256`/`*_SHA256*` literal under `src/birdnet` — so adding or updating a model busts the key while backend edits do not. Declare downloads as quoted literals under one of those names, or the `Compute model cache key` step in `ci.yml` stops tracking them and pins the key to a stale cache; it fails the job if it finds implausibly few.
- Official single-file model downloads (the v3.0 tf/onnx/pt backends) declare a `sha256` in their `ModelInfo`: the download is verified against it once, and the cached file's name carries its first 12 hex chars, so a release that swaps a model under an unchanged family version (`v3.0`) forces a re-download even at an identical byte size. When updating such a model, update URL, sizes and `sha256` together.
- V3.0 species labels are generated, not shipped: one `<lang>.txt` per language, built from each model's own label file plus a taxonomy CSV shared by the acoustic and geo V3.0 models (`utils/taxonomy_v3.py`). The two models join to that taxonomy on different keys deliberately — geo on `species_code`, acoustic on `sci_name` — and a species the taxonomy cannot resolve silently falls back to its English name. Read that module's docstring before changing anything about labels, languages or the taxonomy.
- Runtime logs: one file per inference session, `%TEMP%\birdnet_session_*.log` (Windows) or `/tmp/birdnet_session_*.log`.
