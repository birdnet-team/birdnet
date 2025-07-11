birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv

birdnet-benchmark.exe example/soundscape.wav soundscape.csv
birdnet-benchmark src\\birdnet_v2_debug /tmp/soundscape2.csv -w 1

# raspberry

cd ~
source .venv-bn/bin/activate
pip install birdnet-0.1.7-py3-none-any.whl --force-reinstall

# Birdnet benchmark command line tool

# das geht nicht
# python -m cProfile -o src/birdnet_debug/benchmark_script.prof src/birdnet/benchmark_script.py example/soundscape.wav

python -X importtime src/birdnet/benchmark_script.py example/soundscape.wav 2> src/birdnet_debug/benchmark_script.prof; tuna src/birdnet_debug/benchmark_script.prof

birdnet-benchmark example/soundscape.wav --top-k 6522 --confidence -100
birdnet-benchmark example/soundscape.wav --top-k 10 --confidence -100

birdnet-benchmark example/soundscape.wav --serial-io -w 1
birdnet-benchmark example/soundscape.wav -w 1 --prefetch-ratio 2
birdnet-benchmark example/soundscape.wav -w 1 --prefetch-ratio 3 -p 1

birdnet-benchmark test-dataset/test_dataset_120x60min

birdnet-benchmark example/soundscape.wav test-dataset/test_dataset_1x10min/0.wav --top-k 10 --confidence -100

birdnet-benchmark src/birdnet_debug/audio_formats/soundscape_stereo.wav test-dataset/test_dataset_1x10min/0.wav --top-k 10 --confidence -100

birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv --device GPU --backend pb

birdnet-benchmark test-dataset/test_dataset_4x60min

birdnet-benchmark test-dataset/test_dataset_10000x2min_flac

birdnet-benchmark test-dataset/test_dataset_100000x4s_flac
birdnet-benchmark test-dataset/test_dataset_100000x0.2s_flac
