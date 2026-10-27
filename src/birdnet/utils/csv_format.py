"""Value formatting for the CSV result exports."""

from __future__ import annotations

from typing import Any


def format_input_for_csv(input_value: Any) -> str:  # noqa: ANN401
  # RFC 4180: a quote inside a quoted field is doubled.
  escaped = f"{input_value}".replace('"', '""')
  return f'"{escaped}"'


def hms_centis_fast(v: float) -> str:
  h, rem = divmod(v, 3600)
  m, s = divmod(rem, 60)
  seconds = f"{s:05.2f}"
  if seconds == "60.00":
    # Seconds that round up to a whole minute carry into the minutes.
    seconds = "00.00"
    m += 1
    if m == 60:
      m = 0
      h += 1
  result = f"{int(h):02}:{int(m):02}:{seconds}"
  return result
