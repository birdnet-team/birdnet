birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv

birdnet-benchmark.exe example/soundscape.wav soundscape.csv
birdnet-benchmark src\\birdnet_v2_debug /tmp/soundscape2.csv -w 1

# raspberry

cd ~
source .venv-bn/bin/activate
pip install birdnet-0.1.7-py3-none-any.whl --force-reinstall

# Birdnet benchmark command line tool


birdnet-benchmark example/soundscape.wav /tmp/result.csv --serial-io -w 1
birdnet-benchmark example/soundscape.wav /tmp/result.csv -w 1 --prefetch-ratio 2
birdnet-benchmark example/soundscape.wav /tmp/result.csv -w 1 --prefetch-ratio 3 -p 1

