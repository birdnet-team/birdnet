
birdnet-benchmark example/soundscape.wav /tmp/result.csv --serial-io -w 1

birdnet-benchmark src\\birdnet_v2_debug /tmp/result.csv -w 4 -f 1
#  61 x real-time (RTF: 0.01631669)

birdnet-benchmark src\\birdnet_v2_debug\\60min.wav /tmp/result.csv -w 8 -f 1
#  67 x real-time (RTF: 0.01501786)
