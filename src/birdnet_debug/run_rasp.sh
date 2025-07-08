
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

# Install


kubuntu$ rm -rf dist/; python3.11 -m build -o dist/
kubuntu$ scp dist/birdnet-0.2.0a0-py3-none-any.whl pi@192.168.2.103:/home/pi/birdnet-0.2.0a0-py3-none-any.whl
# scp example/soundscape.wav  pi@192.168.2.103:/home/pi/soundscape.wav

source .venv-bn/bin/activate
pip install /home/pi/birdnet-0.2.0a0-py3-none-any.whl

# Run benchmark

source .venv-bn/bin/activate
birdnet-benchmark /home/pi/soundscape.wav /home/pi/soundscape.csv -w 1


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
