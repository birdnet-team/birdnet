# Review instructions for birdnet

Python library (`src` layout) identifying bird species by sound. General agent
guidance lives in `AGENTS.md`; the checks below are diff-visible rules to apply
when reviewing PRs. Everything here can be verified from the diff text alone.

## Check on every PR

- **Test markers.** New test files must carry the marker matching what they do:
  `load_model` (downloads models), `litert` (imports/exercises `ai_edge_litert` —
  cannot run in the same process after TensorFlow is imported), `gpu`, `fork`
  (forces the fork start method — must run serially), `repro` (needs exact pinned
  versions), `no_tf` (TensorFlow-free surface). A missing or wrong marker breaks
  CI lane isolation even when the test itself passes.
- **Stub sync.** If the diff changes the signature of `load`, `load_custom`, or
  `load_perch_v2` in `src/birdnet/model_loader.py`, it must also update
  `src/birdnet/model_loader.pyi`.
- **Changelog.** User-facing bugfixes and features need a `CHANGELOG.md` entry
  under `[Unreleased]`, written as a self-contained paragraph (cause and effect),
  not just "fixed X".
- **Timing tests.** Latency/duration assertions must test invariants — e.g. the
  minimum of several samples against a floor — never a mean or median, which
  flakes on loaded CI runners.

## Context to avoid false positives

- Red `fork`-lane CI (600 s timeouts in `*_parallel_processes_fork` tests or
  `test_pb_cpu_fork` on macOS) is a known pre-existing race. Before attributing
  it to the PR, check whether the same failure signature occurs on `main`.
- Coverage warnings on fork-only branches are structural: the coverage lane
  cannot exercise every start method. Do not request tests solely to satisfy
  patch coverage there.
- Formatting and typing are enforced by `ruff` and `mypy` (2-space indent, line
  length 88); do not comment on style the tools already govern.
