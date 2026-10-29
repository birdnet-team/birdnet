"""A child's failure slot carries the exception's type and message, bounded."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from birdnet.acoustic.inference.core.sync import (
  FAILURE_SLOT_BYTES,
  clear_failure,
  create_failure_slot,
  read_failure,
  write_failure,
)

pytestmark = pytest.mark.no_tf


class _Foreign(Exception):
  pass


class _Unprintable(Exception):
  def __str__(self) -> str:
    raise ValueError("no text for you")


def test_empty_slot_reads_as_none() -> None:
  assert read_failure(create_failure_slot(mp.get_context("spawn"))) is None


def test_type_and_message_round_trip() -> None:
  slot = create_failure_slot(mp.get_context("spawn"))
  write_failure(slot, RuntimeError("synthetic model failure"))
  assert read_failure(slot) == "RuntimeError: synthetic model failure"


def test_a_foreign_type_is_qualified_by_its_module() -> None:
  slot = create_failure_slot(mp.get_context("spawn"))
  write_failure(slot, _Foreign("bad weights"))
  assert read_failure(slot) == f"{__name__}._Foreign: bad weights"


def test_a_long_message_is_cut_to_the_slot() -> None:
  slot = create_failure_slot(mp.get_context("spawn"))
  write_failure(slot, RuntimeError("x" * (10 * FAILURE_SLOT_BYTES)))
  text = read_failure(slot)
  assert text is not None
  assert text.startswith("RuntimeError: xxx")
  assert len(text.encode()) == FAILURE_SLOT_BYTES - 1


def test_an_exception_whose_str_raises_still_names_its_type() -> None:
  slot = create_failure_slot(mp.get_context("spawn"))
  write_failure(slot, _Unprintable())
  assert read_failure(slot) == f"{__name__}._Unprintable"


def test_clear_empties_the_slot() -> None:
  slot = create_failure_slot(mp.get_context("spawn"))
  write_failure(slot, RuntimeError("x"))
  clear_failure(slot)
  assert read_failure(slot) is None
