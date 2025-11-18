# Consistency Test

## Linux/macOS

Use `and-cuda` extra if you have a CUDA-capable GPU.

```sh
python3.12 -m venv .venv-py12
source .venv-py12/bin/activate
pip install uv
uv pip install birdnet[and-cuda,litert]==0.2.5
python3.12 consistency_test.py
```

## macOS

Use `and-cuda` extra if you have a CUDA-capable GPU.

```sh
python3.12 -m venv .venv-py12
source .venv-py12/bin/activate
pip install uv
uv pip install "birdnet[and-cuda,litert]==0.2.5"
python3.12 consistency_test.py
```

## Windows

```ps1
py -3.12 -m venv .venv-py12
.venv-py12\Scripts\Activate.ps1
pip install uv
uv pip install birdnet==0.2.5
python consistency_test.py
```
