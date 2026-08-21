<!--
  Human or automated tool: the checklist below mirrors what CI actually enforces.
  These are the things that most often bounce a PR here, so it's worth a pass before you push.
  Contributor guidance lives in AGENTS.md and .github/copilot-instructions.md.
-->

## Summary

<!-- What does this change and why? One or two sentences. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / internal (no user-facing behavior change)

## Checklist

- [ ] `ruff check src/birdnet` and `ruff format src/birdnet` are clean, and `mypy` passes (2-space indent, line length 88; type annotations are required).
- [ ] New or changed tests carry the correct marker — `load_model`, `litert`, `gpu`, `fork`, `repro`, `no_tf`, `tf`. A missing or wrong marker breaks CI lane isolation **even when the test itself passes**.
- [ ] If this changes the signature of `load`, `load_custom`, or `load_perch_v2`, the `src/birdnet/model_loader.pyi` stub is updated to match.
- [ ] User-facing fixes/features have a `CHANGELOG.md` entry under `[Unreleased]` — one or two sentences of cause **and** effect, citing this PR as a markdown link.

## Testing

PRs will only be merged if they pass CI.
