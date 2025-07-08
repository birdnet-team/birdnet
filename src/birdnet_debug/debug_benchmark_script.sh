birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv

birdnet-benchmark.exe example/soundscape.wav soundscape.csv
birdnet-benchmark src\\birdnet_v2_debug /tmp/soundscape2.csv -w 1

# raspberry

cd ~
source .venv-bn/bin/activate
pip install birdnet-0.1.7-py3-none-any.whl --force-reinstall

# Birdnet benchmark command line tool

## Install 

## with GPU support
pip install birdnet-0.2.0a0-py3-none-any.whl[and-cuda]

## without GPU support
pip install birdnet-0.2.0a0-py3-none-any.whl

# Example usage of the 

## Predict top 5 species for each segment using CPU und TFLite backend
birdnet-benchmark example/soundscape.wav /tmp/result.csv

## Predict all species
birdnet-benchmark test-dataset/POW /tmp/result.csv --top-k 6522 --confidence -100

## Use protobuf backend
birdnet-benchmark test-dataset/POW /tmp/result.csv -b "pb"

## Run on single GPU
birdnet-benchmark test-dataset/test_dataset_4x60min /tmp/result.csv --backend "pb" --worker 1 --device "GPU" --batch-size 1000

## Run on multiple GPUs
birdnet-benchmark test-dataset/test_dataset_4x60min /tmp/result.csv --backend "pb" --worker 3 --device "GPU:0" "GPU:1" "GPU:2" --batch-size 1000


birdnet-benchmark example/soundscape.wav /tmp/result.csv --serial-io -w 1
birdnet-benchmark example/soundscape.wav /tmp/result.csv -w 1 --prefetch-ratio 2
birdnet-benchmark example/soundscape.wav /tmp/result.csv -w 1 --prefetch-ratio 3 -p 1

