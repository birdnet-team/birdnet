Setup
=====

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation
------------

To install BirdNET, you can use ``pip``:

.. code-block:: bash

  pip install birdnet --user

This will install the latest version of the BirdNET package along with its dependencies.

Ensure reproducibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install BirdNET in a reproducible environment, use the ``repro`` extra, which pins dependency versions and tries to ensure consistent behavior across setups.

.. note::

   Note that it does not include the latest packages, so bugs fixed in newer versions may still be present. This configuration supports CPU execution only and requires Python 3.12. If you have dependencies that overlap with other packages in your system, installations may fail or crash. It is recommended to install BirdNET in a dedicated virtual environment to avoid conflicts.

.. code-block:: bash

  pip install birdnet[repro] --user

CUDA Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install BirdNET with CUDA support, ensure that you have the NVIDIA GPU driver and CUDA installed on your system. Then, use the following command:

.. code-block:: bash

  pip install birdnet[and-cuda] --user

PyTorch and ONNX backends (V3.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The V3.0 acoustic model can additionally be run with the PyTorch (``pt``) and ONNX (``onnx``) backends. These require optional dependencies:

.. code-block:: bash

  # PyTorch backend (.pt models)
  pip install birdnet[pt] --user

  # ONNX backend (.onnx models)
  pip install birdnet[onnx] --user

Both backends support CPU and GPU execution.

Troubleshooting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter issues with audio file reading, please ensure that `libsndfile <https://libsndfile.github.io/libsndfile/>`__ is installed on your system.

- **Ubuntu/Debian**: ``sudo apt-get install libsndfile1``
- **macOS** (using Homebrew): ``brew install libsndfile``
- **Windows**: Download and install the latest precompiled binaries (e.g., ``libsndfile-1.2.2-win64.zip``) from the `official website <https://github.com/libsndfile/libsndfile/releases/>`__, extract them and add the folder to path.

Installation (dev)
------------------

For development purposes, you can clone the repository and install it in editable mode.

Linux & macOS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
  
  # requirement: ensure Python 3.12 is installed
  
  # check out repo
  git clone https://github.com/birdnet-team/birdnet.git
  cd birdnet
  
  # create virtual environment
  python3.12 -m venv .venv312
  source .venv312/bin/activate
  
  # install uv for faster package installation
  python3.12 -m pip install uv
  
  # install package in editable mode with all extra dependencies
  python3.12 -m uv pip install -e .[dev,tests,docs,and-cuda]

Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: ps1

  # requirement: ensure Python 3.12 is installed
  
  # check out repo
  git clone https://github.com/birdnet-team/birdnet.git
  cd birdnet

  # create virtual environment
  py -3.12 -m venv .venv312
  .venv312\\Scripts\\activate

  # install uv for faster package installation
  py -3.12 -m pip install uv
  
  # install package in editable mode with all extra dependencies
  py -3.12 -m uv pip install -e .[dev,tests,docs,and-cuda]

Workflows
-----------

This section describes common workflows for maintaining and using the BirdNET codebase.

Delete cached Python files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To delete all cached Python files from the source code, use the following command:

.. code-block:: bash
  
  find ./src | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf

Run MyPy for type checking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run MyPy for type checking, use the following command:

.. code-block:: bash
  
  # mypy config is in pyproject.toml  
  mypy

Autoformat code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To autoformat the codebase, use the following command:

.. code-block:: bash
  
  autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports ./src/ -r

Run ruff linter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run the ruff linter, use the following command:
  
.. code-block:: bash
  
  # ruff config is in pyproject.toml
  ruff check src/birdnet
  
  # to automatically fix issues
  ruff check src/birdnet --fix

Delete shared memory segments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To delete all shared memory segments (e.g., after a crash), use the following command:

.. code-block:: bash
  
  # list shared memory segments
  ls -lh /dev/shm
  
  # delete all BirdNET shared memory segments, i.e., those starting with "bn_"
  rm /dev/shm/bn_*

Run tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run the test suite, use the following command:

.. code-block:: bash
  
  # requirement: ensure all Python versions (3.11-3.13) are installed
  
  # run all tests for Python 3.11, 3.12, and 3.13
  tox

You can also run only specific tests using the following commands (examples):

.. code-block:: bash
  
  # run only reproducibility tests for Python 3.12
  tox -e py312-repro
  
  # run specific tests with pytest
  pytest -m "not repro and (not litert and not gpu)" -n auto

Calculate code coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To calculate code coverage, use the following command:

.. code-block:: bash
    
  # run all tests with coverage for Python 3.12
  tox -e py312-coverage

Deploy package to PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build and deploy the package to PyPI, use the following commands:

.. code-block:: bash
  
  # requirement 1: ensure code is autoformatted and all tests pass
  # requirement 2: bump version in pyproject.toml, changelog and docs/conf.py
  
  # upgrade build and twine
  python3.12 -m uv pip install --upgrade build twine
  
  # clean previous builds
  rm -rf dist/ build/ *.egg-info/
  
  # build source and wheel distributions to create:
  # -> dist/birdnet-<version>-py3-none-any.whl
  # -> dist/birdnet-<version>.tar.gz
  python3.12 -m build -o dist/
  
  # check built distributions: output should be "PASSED" for both files
  python3.12 -m twine check dist/*
  
  # check that no __pycache__ files are included
  tar -tzf dist/*.tar.gz | grep __pycache__
  
  # upload to PyPI
  python3.12 -m twine upload --repository pypi dist/*

Build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation locally, use the following command:

.. code-block:: bash
  
  # autogenerate doc templates for all modules
  # sphinx-apidoc -o docs/ src/birdnet/
  
  # build HTML docs
  sphinx-build -b html docs/ docs/_build/html -j auto
  
  # OR: autobuild docs after each change
  sphinx-autobuild -b html docs/ docs/_build/html -j auto --watch src/birdnet

Count lines of code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To count the lines of code in the BirdNET source code, use the following command:

.. code-block:: bash
  
  pygount src/birdnet --format=summary

Debugging on Raspberry Pi
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To debug BirdNET on a Raspberry Pi, you can use the following command:

.. code-block:: bash
  
  # copy package and example file to Raspberry Pi
  scp dist/birdnet-<version>-py3-none-any.whl pi@192.168.2.103:/home/pi/birdnet-<version>-py3-none-any.whl
  scp example/soundscape.wav pi@192.168.2.103:/home/pi/soundscape.wav
  
  # setup virtual environment on Raspberry Pi
  ssh pi@192.168.2.103
  sudo apt-get update
  sudo apt-get upgrade
  python3.12 -m venv .venv312
  source .venv312/bin/activate
  python3.12 -m pip install --upgrade pip
  python3.12 -m pip install wheel uv
  
  # remove previous birdnet installation if exists
  python3.12 -m uv pip uninstall birdnet
  
  # install birdnet package from wheel file
  python3.12 -m uv pip install /home/pi/birdnet-<version>-py3-none-any.whl
  
  # run benchmark script
  birdnet-benchmark /home/pi/soundscape.wav --tf-library litert --precision int8

FAQ
---

Where is application data stored?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All BirdNET data (models, benchmarks) is stored in the application-data directory, whose location is platform-specific:

- **Linux:** ``~/.local/share/birdnet``
- **macOS:** ``~/Library/Application Support/birdnet``
- **Windows:** ``%APPDATA%/birdnet``

The default location can be overridden by setting the ``BIRDNET_APP_DATA`` environment variable to any absolute path before the ``birdnet`` package is imported. ::

  # Windows pre-execution script
  set BIRDNET_APP_DATA=C:\Program Files\BirdNET-Analyzer\birdnet-data

  # Linux / macOS pre-execution script
  export BIRDNET_APP_DATA=/opt/birdnet-analyzer/birdnet-data

Why is Python 3.10 not supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python 3.10 is not supported because it is outdated and misses some ``typing`` features, e.g., ``from typing import Self``
