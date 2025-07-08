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

birdnet-benchmark test-dataset/test_dataset_4x60min /tmp/soundscape.csv --device GPU --backend pb

birdnet-benchmark test-dataset/test_dataset_1x1440min /tmp/soundscape.csv --device GPU --backend pb -w 12

# Error
birdnet-benchmark test-dataset/test_dataset_1x1440min /tmp/soundscape.csv --device GPU --backend pb -w 25

birdnet-benchmark test-dataset/test_dataset_100x60min /tmp/soundscape.csv --device GPU --backend pb -w 11 -f 4
# Feeder(s): 4
# Buffer: 21.7/22 filled slots (mean)
# Busy workers: 10.3/11 (mean)
#   Average wait time for next batch: 0.076 ms
# Memory usage:
#   Program: 12695.11 M (total max)
#   Buffer: 12.09 M (shared memory)
#   Result: 2.94 M (NumPy)
# Computational performance:
#   1458 x real-time (RTF: 0.00068592)
# Total performance:
#   1394 x real-time (RTF: 0.00071723)
#   465 segments/s (0:23:14.252536 audio/s)
  

birdnet-benchmark test-dataset/test_dataset_100x60min /tmp/soundscape.csv --device GPU --backend pb -w 1 -f 6 -s 1025
