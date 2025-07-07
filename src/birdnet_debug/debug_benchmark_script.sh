birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv

birdnet-benchmark.exe example/soundscape.wav soundscape.csv

# raspberry

cd ~
source .venv-bn/bin/activate
pip install birdnet-0.1.7-py3-none-any.whl --force-reinstall

birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv
birdnet-benchmark test-dataset/POW /tmp/soundscape.csv -w 2 -d CPU CPU

birdnet-benchmark test-dataset/POW /tmp/soundscape.csv -k 6522 -c -100

birdnet-benchmark test-dataset/POW /tmp/soundscape.csv -b "pb"

