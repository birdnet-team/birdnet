import pytest

from birdnet.core.base import get_session_id_hash
from birdnet_tests.fake_acoustic_backend import colliding_session_ids

pytestmark = pytest.mark.no_tf


def test_the_two_ids_differ_and_share_their_short_hash() -> None:
  first, second = colliding_session_ids()

  assert first != second
  assert get_session_id_hash(first) == get_session_id_hash(second)


def test_every_call_gives_other_ids() -> None:
  # Fixed ids would name the same machine-wide objects in every test run.
  assert set(colliding_session_ids()).isdisjoint(colliding_session_ids())
