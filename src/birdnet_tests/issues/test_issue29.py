"""Replays the reported config from issue #29 (`show_stats="progress"`).

DIAGNOSTIC BUILD for issue #68 — do not merge as-is. This test intermittently
wedges on the macos-15/py3.13 lane until pytest-timeout (600 s) kills the xdist
worker, destroying all evidence: the thread-method stack dump goes to the dead
worker's stderr, and the faulthandler watchdog (armed at timeout + 60 s) never
gets its turn. The `_hang_forensics` fixture below closes that gap: at T+480 s
— while everything is still alive — it snapshots the parent's thread stacks,
every pipeline child's process state, system memory, and (via py-spy, if
installed) the children's Python stacks, into BIRDNET_TEST_WATCHDOG_DIR, which
CI already uploads as an artifact on failure. On a healthy run (~1 s) the probe
never fires and the fixture is inert.
"""

import contextlib
import faulthandler
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import psutil
import pytest

import birdnet

# Snapshot times, all well before pytest-timeout's 600 s process kill. A
# healthy run takes ~1-6 s, so even the first is >20x past normal.
_PROBE_DELAYS_S = (120.0, 300.0, 480.0)
_PY_SPY_TIMEOUT_S = 30


def _dump_py_spy(f, pid: int) -> None:  # noqa: ANN001
  py_spy = shutil.which("py-spy")
  if py_spy is None:
    f.write("py-spy: not installed\n")
    return
  cmd = [py_spy, "dump", "--pid", str(pid)]
  if sys.platform == "darwin":
    # macOS requires elevated rights to read another process; runners have
    # passwordless sudo. -n: never prompt.
    cmd = ["sudo", "-n", *cmd]
  try:
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=_PY_SPY_TIMEOUT_S)
    f.write(res.stdout)
    if res.returncode != 0:
      f.write(f"py-spy rc={res.returncode}: {res.stderr}\n")
  except Exception as e:  # noqa: BLE001 - forensics must never raise
    f.write(f"py-spy failed for pid {pid}: {e!r}\n")


def _write_forensics(path: Path) -> None:
  with path.open("w", encoding="utf-8") as f:
    f.write(f"=== issue #68 forensics @ {datetime.now().isoformat()} ===\n")
    f.write(f"platform={sys.platform} pid={os.getpid()}\n\n")

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    f.write(
      f"memory: total={vm.total / 2**30:.2f}GiB "
      f"available={vm.available / 2**30:.2f}GiB ({vm.percent}% used); "
      f"swap used={sw.used / 2**30:.2f}GiB ({sw.percent}%)\n\n"
    )

    f.write("--- top 10 processes by RSS (system-wide) ---\n")
    procs = []
    for p in psutil.process_iter(["pid", "name", "memory_info"]):
      try:
        procs.append((p.info["memory_info"].rss, p.info["pid"], p.info["name"]))
      except Exception:  # noqa: BLE001, PERF203
        continue
    for rss, pid, name in sorted(procs, reverse=True)[:10]:
      f.write(f"  rss={rss / 2**20:6.0f}MiB pid={pid} {name}\n")

    me = psutil.Process()
    children = me.children(recursive=True)
    f.write(f"\n--- pipeline children of pid {me.pid} ({len(children)}) ---\n")
    for c in children:
      try:
        f.write(
          f"  pid={c.pid} status={c.status()} rss={c.memory_info().rss / 2**20:.0f}MiB "
          f"threads={c.num_threads()} cpu={c.cpu_times().user:.1f}s "
          f"cmd={' '.join(c.cmdline()[:2])!r}\n"
        )
      except Exception as e:  # noqa: BLE001, PERF203
        f.write(f"  pid={c.pid} <gone or unreadable: {e!r}>\n")

    f.write(f"\n--- python thread stacks of test process {me.pid} ---\n")
    f.flush()
    faulthandler.dump_traceback(file=f, all_threads=True)
    f.flush()

    f.write("\n--- py-spy dumps (test process first, then children) ---\n")
    for pid in [me.pid, *(c.pid for c in children)]:
      f.write(f"\n### pid {pid} ###\n")
      f.flush()
      _dump_py_spy(f, pid)
      f.flush()


@pytest.fixture
def _hang_forensics() -> Iterator[None]:
  forensics_dir = os.environ.get("BIRDNET_TEST_WATCHDOG_DIR")
  if not forensics_dir:
    yield
    return

  finished = threading.Event()
  stamp = datetime.now().strftime("%H%M%S")

  def probe() -> None:
    # Multiple snapshots so a wedge can be told apart from a crawl: frozen
    # RSS/CPU numbers between dumps mean a deadlock, moving ones a stall.
    elapsed = 0.0
    for delay_s in _PROBE_DELAYS_S:
      if finished.wait(delay_s - elapsed):
        return  # test finished in time; stay inert
      elapsed = delay_s
      path = (
        Path(forensics_dir)
        / f"forensics_issue29_{os.getpid()}_{stamp}_t{int(delay_s)}s.txt"
      )
      # Diagnostics must never fail or hang the test they observe.
      with contextlib.suppress(Exception):
        _write_forensics(path)

  t = threading.Thread(target=probe, daemon=True, name="issue29-forensics")
  t.start()
  try:
    yield
  finally:
    finished.set()


# Five instances instead of one: each spawns the full default pipeline
# (n_workers=None -> one TF worker per physical core, plus the perf tracker),
# and xdist distributes them across its workers, so several run concurrently.
# That exceeds the contention of the runs where the hang was observed and
# multiplies the trigger chances per CI run.
@pytest.mark.parametrize("run_nr", [1, 2, 3, 4, 5])
@pytest.mark.usefixtures("_hang_forensics")
def test_issue_29(run_nr: int) -> None:
  target = "example/soundscape.wav"
  top_k = None
  batch_size = 1
  prefetch_ratio = 3
  overlap_duration_s = 0.0
  bandpass_fmin = 0
  bandpass_fmax = 15000
  sigmoid_sensitivity = 1.0
  speed = 1.0
  default_confidence_threshold = 0.25
  custom_species_list = None
  progress_callback = None
  show_stats = "progress"
  n_workers = None
  n_producers = 1
  apply_sigmoid = True

  model = birdnet.load(
    "acoustic",
    "2.4",
    "tf",
  )

  model.predict(
    target,
    top_k=top_k,
    batch_size=batch_size,
    prefetch_ratio=prefetch_ratio,
    overlap_duration_s=overlap_duration_s,
    bandpass_fmin=bandpass_fmin,
    bandpass_fmax=bandpass_fmax,
    sigmoid_sensitivity=sigmoid_sensitivity,
    speed=speed,
    default_confidence_threshold=default_confidence_threshold,
    custom_species_list=custom_species_list,
    progress_callback=progress_callback,
    show_stats=show_stats,
    n_workers=n_workers,
    n_producers=n_producers,
    apply_sigmoid=apply_sigmoid,
  )
