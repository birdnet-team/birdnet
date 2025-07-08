# Birdnet benchmark command line tool

## Install 

- with GPU support: `pip install birdnet-0.2.0a0-py3-none-any.whl[and-cuda]`
- without GPU support: `pip install birdnet-0.2.0a0-py3-none-any.whl`

# Example usage

## Predict top 5 species for each segment using CPU und TFLite backend (single file)
birdnet-benchmark soundscape.wav result.csv

## Predict all audio files in a directory
birdnet-benchmark path/to/audio/files/ result.csv

## Use Protobuf backend
birdnet-benchmark soundscape.wav result.csv -b "pb"

## Output predictions for all species
birdnet-benchmark soundscape.wav result.csv --top-k 6522 --confidence -100

## Run on single GPU
birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 1 --device "GPU" --batch-size 1000

## Run on multiple GPUs
birdnet-benchmark soundscape.wav /tmp/result.csv --backend "pb" --worker 3 --device "GPU:0" "GPU:1" "GPU:2" --batch-size 1000

## Increase amount of data feeders
birdnet-benchmark soundscape.wav /tmp/result.csv --feeders 2
