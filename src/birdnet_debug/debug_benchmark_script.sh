birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv

birdnet-benchmark.exe example/soundscape.wav soundscape.csv
birdnet-benchmark src\\birdnet_v2_debug /tmp/soundscape2.csv -w 1

# raspberry

cd ~
source .venv-bn/bin/activate
pip install birdnet-0.1.7-py3-none-any.whl --force-reinstall

# Birdnet benchmark command line tool

birdnet-benchmark example/soundscape.wav /tmp/result.csv --top-k 6522 --confidence -100
birdnet-benchmark example/soundscape.wav /tmp/result.csv --top-k 10 --confidence -100

birdnet-benchmark example/soundscape.wav /tmp/result.csv --serial-io -w 1
birdnet-benchmark example/soundscape.wav /tmp/result.csv -w 1 --prefetch-ratio 2
birdnet-benchmark example/soundscape.wav /tmp/result.csv -w 1 --prefetch-ratio 3 -p 1


birdnet-benchmark example/soundscape.wav test-dataset/test_dataset_1x10min/0.wav /tmp/result.csv --top-k 10 --confidence -100

birdnet-benchmark src/birdnet_debug/audio_formats/soundscape_stereo.wav test-dataset/test_dataset_1x10min/0.wav /tmp/result.csv --top-k 10 --confidence -100

birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv --device GPU --backend pb