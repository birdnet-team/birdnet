
birdnet-benchmark example/soundscape.wav --serial-io -w 1
birdnet-benchmark example/soundscape.wav

birdnet-benchmark src\\birdnet_v2_debug -w 4 -f 1
#  61 x real-time (RTF: 0.01631669)

birdnet-benchmark src\\birdnet_v2_debug\\60min.wav -w 8 -f 1
#  67 x real-time (RTF: 0.01501786)

birdnet-benchmark test-dataset\\test_dataset_10x60min_wav -w 4 -f 1

pip install build
python.exe -m build -o dist/

# CMD
"C:\Program Files\Python311\python.exe" -m venv .venv-bn
.venv-bn\\Scripts\\activate
python.exe -m pip install --upgrade pip
python.exe -m pip install wheel
python.exe -m pip install ..\\dist\\birdnet-0.2.0a0-py3-none-any.whl