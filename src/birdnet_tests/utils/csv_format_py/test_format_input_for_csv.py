import pytest

from birdnet.utils.csv_format import format_input_for_csv

pytestmark = pytest.mark.no_tf


def test_format_input_for_csv() -> None:
  assert format_input_for_csv("bird") == '"bird"'
  assert format_input_for_csv(42) == '"42"'


def test_format_input_for_csv_doubles_quotes() -> None:
  assert format_input_for_csv('a"b') == '"a""b"'
  assert format_input_for_csv('"') == '""""'
