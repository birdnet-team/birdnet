birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv

birdnet-benchmark.exe example/soundscape.wav soundscape.csv
birdnet-benchmark src\\birdnet_v2_debug /tmp/soundscape2.csv -w 1

# raspberry

cd ~
source .venv-bn/bin/activate
pip install birdnet-0.1.7-py3-none-any.whl --force-reinstall

# Birdnet benchmark command line tool

birdnet-benchmark example/soundscape.wav --top-k 6522 --confidence -100
birdnet-benchmark example/soundscape.wav --top-k 10 --confidence -100

birdnet-benchmark example/soundscape.wav --serial-io -w 1
birdnet-benchmark example/soundscape.wav -w 1 --prefetch-ratio 2
birdnet-benchmark example/soundscape.wav -w 1 --prefetch-ratio 3 -p 1


birdnet-benchmark example/soundscape.wav test-dataset/test_dataset_1x10min/0.wav --top-k 10 --confidence -100

birdnet-benchmark src/birdnet_debug/audio_formats/soundscape_stereo.wav test-dataset/test_dataset_1x10min/0.wav --top-k 10 --confidence -100

birdnet-benchmark example/soundscape.wav /tmp/soundscape.csv --device GPU --backend pb

birdnet-benchmark test-dataset/test_dataset_4x60min /tmp/soundscape.csv --device GPU --backend pb

birdnet-benchmark test-dataset/test_dataset_1x1440min /tmp/soundscape.csv --device GPU --backend pb -w 12

# Error
birdnet-benchmark test-dataset/test_dataset_1x1440min --device GPU --backend pb -w 25

birdnet-benchmark test-dataset/test_dataset_100x60min --device GPU --backend pb -w 11 -f 4
# Wall time:  0:04:18.202865
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
  
# Batchsize max. 1025 on Titan RTX
birdnet-benchmark test-dataset/test_dataset_100x60min --device GPU --backend pb -w 1 -f 5 -s 1025

birdnet-benchmark test-dataset/test_dataset_100x60min_flac --device GPU --backend pb -w 1 -f 5 -s 1025


birdnet-benchmark test-dataset/test_dataset_100x60min -w 8 -f 5

birdnet-benchmark test-dataset/test_dataset_100x60min --device GPU --backend pb -w 8 -f 5 --confidence "-1" --top-k 1


birdnet-benchmark /data/datasets/l2arctic --device GPU --backend pb -w 1 -f 6 -s 1025 --confidence "-1" --top-k 1

# Wall time:  0:01:28.809129
# Input: 26889 file(s) (WAV)
#   Total duration: 1 day, 3:30:31.551406
#   Average duration: 0:00:03.682976
#   Minimum duration (single file): 0:00:00.680000
#   Maximum duration (single file): 0:03:55.036417
# Feeder(s): 6
# Buffer: 1.7/2 filled slots (mean)
# Busy workers: 0.9/1 (mean)
#   Average wait time for next batch: 50.601 ms
# Memory usage:
#   Program: 11860.31 M (total max)
#   Buffer: 1126.11 M (shared memory)
#   Result: 56.13 M (NumPy)
# Computational performance:
#   1742 x real-time (RTF: 0.00057389)
# Total performance:
#   1561 x real-time (RTF: 0.00064069)
#   520 segments/s (0:26:00.819273 audio/s)
  
birdnet-benchmark /data/datasets/LJSpeech-1.1 /tmp/ljs.csv --device GPU --backend pb -w 1 -f 6 -s 1025
# Wall time:  0:03:19.381935
# Input: 13100 file(s) (WAV)
#   Total duration: 23:55:17.076281
#   Average duration: 0:00:06.573823
#   Minimum duration (single file): 0:00:01.110068
#   Maximum duration (single file): 0:00:10.096190
# Feeder(s): 6
# Buffer: 0.6/2 filled slots (mean)
# Busy workers: 0.3/1 (mean)
#   Average wait time for next batch: 3481.062 ms
# Memory usage:
#   Program: 11117.34 M (total max)
#   Buffer: 1126.11 M (shared memory)
#   Result: 3.65 M (NumPy)
# Computational performance:
#   551 x real-time (RTF: 0.00181538)
# Total performance:
#   532 x real-time (RTF: 0.00187795)
#   177 segments/s (0:08:52.495585 audio/s)
