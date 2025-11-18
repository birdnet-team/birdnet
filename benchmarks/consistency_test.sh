
# Linux/macOS

```sh
python3.12 -m venv .venv-py12
source .venv-py12/bin/activate
pip install birdnet[and-cuda,litert]
python3.12 consistency_test.py
```

# Windows

```ps1
py -311 -m venv .venv-py12
.venv-py12\Scripts\Activate.ps1
pip install birdnet
python consistency_test.py
```
