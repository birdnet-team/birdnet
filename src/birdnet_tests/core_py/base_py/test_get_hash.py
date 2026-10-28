from birdnet.core.base import get_hash


def test_get_hash_is_deterministic() -> None:
  assert get_hash("session") == get_hash("session")
  assert get_hash("session") != get_hash("other")
  # sha256 hex digest is 64 characters
  assert len(get_hash("session")) == 64
