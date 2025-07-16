
# Setup

# Raspberry PI
sudo apt install rpi-imager

ssh pi@192.168.2.103

## Auf dem Raspberry PI
sudo apt-get update
sudo apt-get upgrade
python3.11 -m venv .venv-bn
source .venv-bn/bin/activate
pip install --upgrade pip
pip install wheel

# Install / Update

## First on Kubuntu
rm -rf dist/; python3.11 -m build -o dist/
scp dist/birdnet-0.2.0a0-py3-none-any.whl pi@192.168.2.103:/home/pi/birdnet-0.2.0a0-py3-none-any.whl
scp example/soundscape.wav  pi@192.168.2.103:/home/pi/soundscape.wav
scp test-dataset/test_dataset_1x10min/0.wav  pi@192.168.2.103:/home/pi/10min.wav
scp test-dataset/test_dataset_1x60min/0.wav  pi@192.168.2.103:/home/pi/60min.wav

## Then on Pi
source .venv-bn/bin/activate
pip uninstall birdnet -y; pip install /home/pi/birdnet-0.2.0a0-py3-none-any.whl

# Run benchmark

source .venv-bn/bin/activate
birdnet-benchmark /home/pi/soundscape.wav 
birdnet-benchmark /home/pi/soundscape.wav -w 1 --prefetch-ratio 1 --feeders 1
birdnet-benchmark /home/pi/10min.wav -w 1 --prefetch-ratio 1 --feeders 1
birdnet-benchmark /home/pi/60min.wav -w 2 --prefetch-ratio 1 --feeders 1 #   7 x real-time (RTF: 0.15280223)
birdnet-benchmark /home/pi/60min.wav -w 2 --prefetch-ratio 1 --feeders 1 -p int8


# Sonstiges

Build LiteRT Python Wheel Package

url="git@github.com:birdnet-team/birdnet.git"

mkdir /home/pi/code/
cd /home/pi/code/

git clone "$url" \
  --config core.sshCommand="ssh -i ~/.ssh/id_rsa" \
  --config user.name="Stefan Taubert" \
  --config user.email="23339395+stefantaubert@users.noreply.github.com"
chmod 600 /home/pi/.ssh/id_rsa
cd birdnet/
git checkout refactor-the-whole-project 
# kein Pull nötig


source .venv-py311/bin/activate

