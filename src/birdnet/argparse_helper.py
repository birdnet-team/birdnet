import argparse
import codecs
import json
from argparse import ArgumentTypeError
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TypeVar

from ordered_set import OrderedSet

T = TypeVar("T")


class ConvertToOrderedSetAction(argparse._StoreAction):
  def __call__(
    self,
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    values: list | None,
    option_string: str | None = None,
  ):
    if values is not None:
      values = OrderedSet(values)
    super().__call__(parser, namespace, values, option_string)


class ConvertToSetAction(argparse._StoreAction):
  def __call__(
    self,
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
    values: list | None,
    option_string: str | None = None,
  ):
    if values is not None:
      values = set(values)
    super().__call__(parser, namespace, values, option_string)


def parse_json(value: str) -> dict:
  path = parse_path(value)
  try:
    with open(path, encoding="utf-8") as file:
      result = json.load(file)
  except Exception as ex:
    raise ArgumentTypeError("JSON couldn't be parsed!") from ex
  return result


def parse_codec(value: str) -> str:
  value = parse_required(value)
  try:
    codecs.lookup(value)
  except LookupError as error:
    raise ArgumentTypeError("Codec was not found!") from error
  return value


def parse_path(value: str) -> Path:
  value = parse_required(value)
  try:
    path = Path(value)
  except ValueError as error:
    raise ArgumentTypeError("Value needs to be a path!") from error
  return path


def parse_optional_value(value: str, method: Callable[[str], T]) -> T | None:
  if value is None:
    return None
  return method(value)


def get_optional(method: Callable[[str], T]) -> Callable[[str], T | None]:
  result = partial(
    parse_optional_value,
    method=method,
  )
  return result


def parse_existing_file(value: str) -> Path:
  path = parse_path(value)
  if not path.is_file():
    raise ArgumentTypeError("File was not found!")
  return path


def parse_existing_directory(value: str) -> Path:
  path = parse_path(value)
  if not path.is_dir():
    raise ArgumentTypeError("Directory was not found!")
  return path


def parse_required(value: str | None) -> str:
  if value is None:
    raise ArgumentTypeError("Value must not be None!")
  return value


def parse_non_empty(value: str | None) -> str:
  value = parse_required(value)
  if value == "":
    raise ArgumentTypeError("Value must not be empty!")
  return value


def parse_non_empty_or_whitespace(value: str) -> str:
  value = parse_required(value)
  if value.strip() == "":
    raise ArgumentTypeError("Value must not be empty or whitespace!")
  return value


def parse_float(value: str) -> float:
  value = parse_required(value)
  try:
    value = float(value)
  except ValueError as error:
    raise ArgumentTypeError("Value needs to be a decimal number!") from error
  return value


def parse_percent(value: str) -> float:
  value = parse_float(value)
  if not (0 <= value <= 100):
    raise ArgumentTypeError("Value needs to be in range [0, 100]!")
  return value


def parse_positive_float(value: str) -> float:
  value = parse_float(value)
  if not value > 0:
    raise ArgumentTypeError("Value needs to be greater than zero!")
  return value


def parse_non_negative_float(value: str) -> float:
  value = parse_float(value)
  if not value >= 0:
    raise ArgumentTypeError("Value needs to be greater than or equal to zero!")
  return value


def parse_integer(value: str) -> int:
  value = parse_required(value)
  if not value.isdigit():
    raise ArgumentTypeError("Value needs to be an integer!")
  value = int(value)
  return value


def parse_positive_integer(value: str) -> int:
  value = parse_integer(value)
  if not value > 0:
    raise ArgumentTypeError("Value needs to be greater than zero!")
  return value


def parse_integer_greater_one(value: str) -> int:
  value = parse_integer(value)
  if not value > 1:
    raise ArgumentTypeError("Value needs to be greater than one!")
  return value


def parse_float_greater_one(value: str) -> int:
  value = parse_float(value)
  if not value > 1:
    raise ArgumentTypeError("Value needs to be greater than one!")
  return value


def parse_non_negative_integer(value: str) -> int:
  value = parse_integer(value)
  if not value >= 0:
    raise ArgumentTypeError("Value needs to be greater than or equal to zero!")
  return value


def parse_datetime(value: str) -> datetime:
  value = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
  return value
