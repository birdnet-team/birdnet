# Contributing

If you notice an error, please don't hesitate to open an issue.

## Development setup

```sh
# update
sudo apt update
# install Python 3.9-3.11 for ensuring that tests can be run
sudo apt install python3-pip \
  python3.9 python3.9-dev python3.9-distutils python3.9-venv \
  python3.10 python3.10-dev python3.10-distutils python3.10-venv \
  python3.11 python3.11-dev python3.11-distutils python3.11-venv
# install pipenv for creation of virtual environments
python3.8 -m pip install pipenv --user

# check out repo
git clone https://github.com/birdnet-team/birdnet.git
cd birdnet
# create virtual environment
python3.8 -m pipenv install --dev
```

## Running the tests

```sh
# first install the tool like in "Development setup"
# then, navigate into the directory of the repo (if not already done)
cd birdnet
# activate environment
python3.8 -m pipenv shell
# run tests
tox
```

Final lines of test result output:

```log
  py39: commands succeeded
  py310: commands succeeded
  py311: commands succeeded
  congratulations :)
```
