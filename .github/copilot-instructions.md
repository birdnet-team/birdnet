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
- **Cross-instance result comparisons.** A test comparing results produced by
  *different* interpreter instances — separate sessions, separate processes,
  separate threads, or `n_workers > 1` — must use a tolerance
  (`assert_prediction_result_is_close`), never exact float equality. Two
  interpreters need not agree to the last bit, because the thread pool is sized
  from the visible cores and that changes the reduction order. Exact equality is
  correct only when a single interpreter produced both results (one session,
  `n_workers=1`).
- **V3.0 taxonomy and language changes.** A diff touching `_LANGUAGE_TO_COLUMN` in
  `acoustic/models/v3_0/model.py` or `geo/models/v3_0/model.py` must move the
  language lists in `globals.py` (`MODEL_LANGUAGES_V3_0`,
  `VALID_MODEL_LANGUAGES_V3_0`) with it — both models share them. A diff changing
  the taxonomy URL or size in `utils/taxonomy_v3.py` must keep every mapped column
  available in the new file: a missing column does not raise, it silently yields a
  complete file of English names.
- **Tests for a hang.** A test whose failure mode is the pipeline not returning
  must run the session in a worker thread behind a deadline, so a regression
  fails with a readable message instead of wedging until the suite timeout.

## Context to avoid false positives

- A red `fork` lane on Linux or Windows is *not* expected any more. The
  intermittent 600 s wedges there were fixed by moving the ring-buffer attach out
  of the fork children (#67), and the lane has been green since. Treat one as a
  real finding rather than known noise.
- The exception is macOS `test_pb_cpu_fork`, which can still die in a native
  TensorFlow crash inside a forked child. That is the fork-after-TensorFlow
  limitation `birdnet_tests/conftest.py` documents, not something a PR caused.
- Coverage warnings on fork-only branches are structural: the coverage lane
  cannot exercise every start method. Do not request tests solely to satisfy
  patch coverage there.
- Formatting and typing are enforced by `ruff` and `mypy` (2-space indent, line
  length 88); do not comment on style the tools already govern.
