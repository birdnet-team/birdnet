"""The tox lanes partition the suite by marker, serial markers in serial lanes.

Checked on the lane expressions in ``pyproject.toml`` alone: a test in the wrong
lane shows up as a hang on a GPU machine, which CI does not have.
"""

from __future__ import annotations

import configparser
import itertools
import shlex
import tomllib
from pathlib import Path
from typing import NamedTuple

import pytest

from birdnet_tests import conftest

pytestmark = pytest.mark.no_tf

_PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"
_MARKERS = ("load_model", "litert", "gpu", "fork", "no_tf")
_COMBINATIONS = [
  frozenset(combination)
  for size in range(len(_MARKERS) + 1)
  for combination in itertools.combinations(_MARKERS, size)
]
_TF_ENVS = {"testenv", "testenv:py312-repro", "testenv:py312-coverage"}
_TF_FREE_ENVS = {"testenv:py314", "testenv:py315", "testenv:py313-notf"}


class _Lane(NamedTuple):
  expression: str
  workers: str

  def selects(self, combination: frozenset[str]) -> bool:
    # Private to pytest, so imported here: a release that moves it fails these
    # tests, not the collection of every lane.
    from _pytest.mark.expression import Expression

    return Expression.compile(self.expression).evaluate(
      lambda name: name in combination
    )

  def __str__(self) -> str:
    return f"'{self.expression}' (-n {self.workers})"


def _option(argv: list[str], flag: str, section: str) -> str:
  assert flag in argv, f"[{section}] '{shlex.join(argv)}' has no {flag}"
  return argv[argv.index(flag) + 1]


def _lanes() -> dict[str, list[_Lane]]:
  tox_ini = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["tool"]["tox"][
    "legacy_tox_ini"
  ]
  parser = configparser.ConfigParser(interpolation=None)
  parser.read_string(tox_ini)
  lanes: dict[str, list[_Lane]] = {}
  for section in parser.sections():
    if not section.startswith("testenv"):
      continue
    for command in parser.get(section, "commands", fallback="").splitlines():
      if not command.strip().startswith("pytest "):
        continue
      argv = shlex.split(command)
      lanes.setdefault(section, []).append(
        _Lane(_option(argv, "-m", section), _option(argv, "-n", section))
      )
  return lanes


def _in_domain(section: str, combination: frozenset[str]) -> bool:
  if section in _TF_FREE_ENVS:
    return bool(combination & {"no_tf", "litert"}) and not (
      combination & {"fork", "gpu"}
    )
  if section == "testenv:py312-repro" and "gpu" in combination:
    return False
  # Policy: no lane serves a litert test that also needs a GPU.
  return not {"litert", "gpu"} <= combination


def _name(combination: frozenset[str]) -> str:
  return "+".join(m for m in _MARKERS if m in combination) or "(unmarked)"


def _selections() -> list[tuple[str, _Lane, frozenset[str]]]:
  return [
    (section, lane, combination)
    for section, lanes in _lanes().items()
    for lane in lanes
    for combination in _COMBINATIONS
    if lane.selects(combination)
  ]


def test_every_environment_with_pytest_lanes_is_classified() -> None:
  assert set(_lanes()) == _TF_ENVS | _TF_FREE_ENVS


def test_every_marker_combination_runs_in_exactly_one_lane() -> None:
  problems = []
  for section, lanes in _lanes().items():
    for combination in _COMBINATIONS:
      selected_by = [str(lane) for lane in lanes if lane.selects(combination)]
      expected = 1 if _in_domain(section, combination) else 0
      if len(selected_by) != expected:
        problems.append(
          f"[{section}] {_name(combination)}: expected in {expected} lane(s), "
          f"selected by {len(selected_by)}: {', '.join(selected_by) or 'none'}"
        )

  assert not problems, "\n" + "\n".join(problems)


def test_every_collected_test_has_a_lane_with_tensorflow() -> None:
  # -m hides a test that no lane selects, so this asks what was collected.
  problems = sorted(
    f"{nodeid} is {_name(markers & frozenset(_MARKERS))}"
    for nodeid, markers in conftest.collected_markers.items()
    if not _in_domain("testenv", markers & frozenset(_MARKERS))
  )

  assert not problems, "\n" + "\n".join(problems)


def test_fork_tests_run_in_process() -> None:
  problems = [
    f"[{section}] {_name(combination)} is selected by {lane}"
    for section, lane, combination in _selections()
    if "fork" in combination and lane.workers != "0"
  ]

  assert not problems, "\n" + "\n".join(problems)


def test_gpu_tests_run_one_at_a_time() -> None:
  problems = [
    f"[{section}] {_name(combination)} is selected by {lane}"
    for section, lane, combination in _selections()
    if "gpu" in combination and lane.workers not in ("0", "1")
  ]

  assert not problems, "\n" + "\n".join(problems)


def test_a_lane_with_tensorflow_is_all_litert_or_has_none() -> None:
  # ai_edge_litert cannot be imported after TensorFlow.
  problems = []
  for section, lanes in _lanes().items():
    if section not in _TF_ENVS:
      continue
    for lane in lanes:
      selected = [c for c in _COMBINATIONS if lane.selects(c)]
      litert = [c for c in selected if "litert" in c]
      others = [c for c in selected if "litert" not in c]
      if litert and others:
        problems.append(
          f"[{section}] {lane} selects {', '.join(map(_name, litert))} "
          f"beside {', '.join(map(_name, others))}"
        )

  assert not problems, "\n" + "\n".join(problems)


def test_downloads_run_no_later_than_the_tests_they_serve() -> None:
  # A test's files come from the load_model tests with no marker it lacks.
  problems = []
  for section, lanes in _lanes().items():
    lane_of = {
      combination: index
      for index, lane in enumerate(lanes)
      for combination in _COMBINATIONS
      if lane.selects(combination)
    }
    for user, user_lane in lane_of.items():
      if "load_model" in user:
        continue
      for downloads, download_lane in lane_of.items():
        if (
          "load_model" in downloads
          and downloads - {"load_model"} <= user
          and download_lane > user_lane
        ):
          problems.append(
            f"[{section}] {_name(user)} runs in {lanes[user_lane]}, before "
            f"{_name(downloads)} in {lanes[download_lane]}"
          )

  assert not problems, "\n" + "\n".join(problems)
