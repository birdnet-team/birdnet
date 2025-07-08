# Birdnet benchmark command line tool

## Install 

Preparation on Windows (CMD):

```cmd
py -3.11 -m venv .venv-bn
.venv-bn\\Scripts\\activate
python.exe -m pip install --upgrade pip
python.exe -m pip install wheel
python.exe -m pip install birdnet-0.2.0a0-py3-none-any.whl
```

Preparation on Linux (Bash):

```sh
python3.11 -m venv .venv-bn
source .venv-bn/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install birdnet-0.2.0a0-py3-none-any.whl
```

- install with GPU support: `pip install birdnet-0.2.0a0-py3-none-any.whl[and-cuda]`

## Example usage

### Show benchmark options

`birdnet-benchmark --help`

### Predict top 5 species for each segment using CPU und TFLite backend (single file)

`birdnet-benchmark soundscape.wav result.csv`

### Predict all audio files in a directory

`birdnet-benchmark path/to/audio/files/ result.csv`

### Use Protobuf backend

`birdnet-benchmark soundscape.wav result.csv -b "pb"`

### Output predictions for top 10 species

`birdnet-benchmark soundscape.wav result.csv --top-k 10 --confidence -100`

### Run on single GPU

`birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 1 --device "GPU" --batch-size 1000`

### Run on multiple GPUs

`birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 3 --device "GPU:0" "GPU:1" "GPU:2" --batch-size 1000`

### Increase amount of data feeders

`birdnet-benchmark soundscape.wav /tmp/result.csv --feeders 2`
