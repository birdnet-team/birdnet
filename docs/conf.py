# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import tomllib
from pathlib import Path

project = "birdnet"
copyright = "2026, Stefan Taubert"
author = "Stefan Taubert"
# Read straight from pyproject.toml (the single source of truth) so the docs
# version never drifts and needs no manual bump. Works in editable dev and in
# CI alike, without depending on the installed package metadata.
_pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
release = tomllib.loads(_pyproject.read_text(encoding="utf-8"))["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.autosummary",
  "sphinx_autodoc_typehints",
  "sphinx.ext.napoleon",
  "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

autosummary_generate = True

# Google-style docstrings (Args:/Returns:) are converted to reST by napoleon.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

typehints_fully_qualified = False
