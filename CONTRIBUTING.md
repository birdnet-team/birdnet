# Contributing

If you notice a bug or have an idea, please open an issue using the templates.
Pull requests are welcome — the notes below get a development environment running
and match what CI enforces.

## Development setup

`birdnet` supports Python 3.11–3.14 (3.11 is the minimum; 3.10 lacks
`typing.Self`). CI runs the test matrix on all four, but you only need the
versions you intend to test against — `tox` skips interpreters it can't find.

Audio loading needs the `libsndfile` system library:

- Debian/Ubuntu: `sudo apt install libsndfile1`
- macOS: `brew install libsndfile`
- Windows: `choco install libsndfile`

Then set up the environment (Debian/Ubuntu example; the deadsnakes PPA provides
the interpreters):

```sh
sudo apt update
sudo apt install python3-pip python3.11 python3.11-dev python3.11-venv libsndfile1

git clone https://github.com/birdnet-team/birdnet.git
cd birdnet
python3.11 -m venv .venv311
source .venv311/bin/activate
python3.11 -m pip install uv

# TensorFlow is optional; [tf] enables the default TFLite/SavedModel backends and
# [pt] the torch backend. onnxruntime is already a base dependency. On a Linux
# machine with a GPU, use [and-cuda] in place of [tf].
uv pip install -e '.[tests,dev,tf,pt]'
```

## Checks

Formatting, linting and typing are enforced by CI. Run them before pushing:

```sh
ruff format src/birdnet
ruff check src/birdnet
mypy
```

## Running the tests

For day-to-day work, the fast subset skips model downloads and the isolated lanes:

```sh
pytest -m "not repro and not load_model and not litert and not gpu and not fork" -n auto
```

To run the full matrix exactly as CI does (needs the Python 3.11–3.14 interpreters
installed):

```sh
tox
```

Official models auto-download on first use (~3 GB for the full set). Set the
`BIRDNET_APP_DATA` environment variable to a persistent path to keep the cache
between runs. `AGENTS.md` documents the test markers (`load_model`, `litert`,
`gpu`, `fork`, `repro`, `no_tf`, `tf`) and why their run order matters.

## Pull requests

User-facing fixes and features need a `CHANGELOG.md` entry under `[Unreleased]`.
The pull request template lists the full checklist CI expects.
