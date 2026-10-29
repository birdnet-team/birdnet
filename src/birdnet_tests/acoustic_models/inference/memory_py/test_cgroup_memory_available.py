"""The cgroup memory limit is read from the tightest ancestor, v2 or v1."""

from __future__ import annotations

from pathlib import Path

import pytest

from birdnet.acoustic.inference.memory import cgroup_memory_available

pytestmark = pytest.mark.no_tf

_GIB = 2**30


def _v2(root: Path, rel: str, **files: str) -> None:
  d = root / rel
  d.mkdir(parents=True, exist_ok=True)
  for name, text in files.items():
    (d / name.replace("_", ".", 1)).write_text(text)


def _proc(tmp_path: Path, text: str) -> Path:
  p = tmp_path / "proc_self_cgroup"
  p.write_text(text)
  return p


def test_v2_limit_on_an_ancestor_minus_its_anon_usage(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  _v2(
    root,
    "a",
    memory_max=str(4 * _GIB),
    memory_current=str(2 * _GIB),
    memory_stat=f"anon {_GIB}\nfile {_GIB}\n",
  )
  _v2(root, "a/b", memory_max="max", memory_current=str(_GIB), memory_stat="file 0\n")
  proc = _proc(tmp_path, "0::/a/b\n")
  # 4 GiB limit, 2 GiB current of which 1 GiB is reclaimable page cache.
  assert cgroup_memory_available(root, proc) == 3 * _GIB


def test_v2_tightest_of_several_limits_wins(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  _v2(root, "a", memory_max=str(8 * _GIB), memory_current="0", memory_stat="file 0\n")
  _v2(
    root,
    "a/b",
    memory_max=str(2 * _GIB),
    memory_current=str(_GIB),
    memory_stat="file 0\n",
  )
  proc = _proc(tmp_path, "0::/a/b\n")
  assert cgroup_memory_available(root, proc) == _GIB


def test_v2_root_cgroup_of_a_container_namespace(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  _v2(root, ".", memory_max=str(_GIB), memory_current="0", memory_stat="file 0\n")
  proc = _proc(tmp_path, "0::/\n")
  assert cgroup_memory_available(root, proc) == _GIB


def test_v2_without_a_limit_is_none(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  _v2(root, "a/b", memory_max="max", memory_current="5")
  proc = _proc(tmp_path, "0::/a/b\n")
  assert cgroup_memory_available(root, proc) is None


def test_usage_above_the_limit_reads_as_nothing_left(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  _v2(
    root,
    "a",
    memory_max=str(_GIB),
    memory_current=str(2 * _GIB),
    memory_stat="file 0\n",
  )
  proc = _proc(tmp_path, "0::/a\n")
  assert cgroup_memory_available(root, proc) == 0


def test_v1_limit_minus_usage_plus_cache(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  d = root / "memory" / "docker" / "abc"
  d.mkdir(parents=True)
  (d / "memory.limit_in_bytes").write_text(str(4 * _GIB))
  (d / "memory.usage_in_bytes").write_text(str(3 * _GIB))
  (d / "memory.stat").write_text(f"cache {_GIB}\nrss {2 * _GIB}\ntotal_cache {_GIB}\n")
  proc = _proc(tmp_path, "12:memory:/docker/abc\n11:cpu,cpuacct:/docker/abc\n")
  assert cgroup_memory_available(root, proc) == 2 * _GIB


def test_v1_unlimited_value_is_none(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  d = root / "memory"
  d.mkdir(parents=True)
  (d / "memory.limit_in_bytes").write_text("9223372036854771712")
  (d / "memory.usage_in_bytes").write_text("1")
  proc = _proc(tmp_path, "12:memory:/\n")
  assert cgroup_memory_available(root, proc) is None


def test_unreadable_proc_file_is_none(tmp_path: Path) -> None:
  assert cgroup_memory_available(tmp_path, tmp_path / "missing") is None


def test_this_machine_reads_without_raising() -> None:
  value = cgroup_memory_available()
  assert value is None or value >= 0


def test_v2_path_absent_from_the_mount_reads_the_mount_root(tmp_path: Path) -> None:
  # Docker without a private cgroup namespace: the mount is the container's
  # own cgroup, while /proc/self/cgroup still names the host-side path.
  root = tmp_path / "sys"
  _v2(
    root,
    ".",
    memory_max=str(2 * _GIB),
    memory_current=str(_GIB),
    memory_stat="file 0\n",
  )
  proc = _proc(tmp_path, "0::/docker/abc\n")
  assert cgroup_memory_available(root, proc) == _GIB


def test_v2_shared_memory_is_not_reclaimable(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  _v2(
    root,
    "a",
    memory_max=str(4 * _GIB),
    memory_current=str(3 * _GIB),
    memory_stat=f"file {2 * _GIB}\nshmem {2 * _GIB}\n",
  )
  proc = _proc(tmp_path, "0::/a\n")
  assert cgroup_memory_available(root, proc) == _GIB


def test_v2_unreadable_usage_never_reports_more_than_the_limit(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  _v2(root, "a", memory_max=str(_GIB), memory_stat=f"file {_GIB // 2}\n")
  proc = _proc(tmp_path, "0::/a\n")
  assert cgroup_memory_available(root, proc) == _GIB


def test_v1_path_absent_from_the_mount_reads_the_mount_root(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  d = root / "memory"
  d.mkdir(parents=True)
  (d / "memory.limit_in_bytes").write_text(str(4 * _GIB))
  (d / "memory.usage_in_bytes").write_text(str(_GIB))
  (d / "memory.stat").write_text("total_cache 0\n")
  proc = _proc(tmp_path, "12:memory:/docker/abc\n")
  assert cgroup_memory_available(root, proc) == 3 * _GIB


def test_v1_hierarchical_limit_of_an_ancestor_counts(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  d = root / "memory" / "a"
  d.mkdir(parents=True)
  (d / "memory.limit_in_bytes").write_text("9223372036854771712")
  (d / "memory.usage_in_bytes").write_text(str(_GIB))
  (d / "memory.stat").write_text(
    f"total_cache 0\nhierarchical_memory_limit {3 * _GIB}\n"
  )
  proc = _proc(tmp_path, "12:memory:/a\n")
  assert cgroup_memory_available(root, proc) == 2 * _GIB


def test_v1_shared_memory_is_not_reclaimable(tmp_path: Path) -> None:
  root = tmp_path / "sys"
  d = root / "memory"
  d.mkdir(parents=True)
  (d / "memory.limit_in_bytes").write_text(str(4 * _GIB))
  (d / "memory.usage_in_bytes").write_text(str(3 * _GIB))
  (d / "memory.stat").write_text(f"total_cache {2 * _GIB}\ntotal_shmem {2 * _GIB}\n")
  proc = _proc(tmp_path, "12:memory:/\n")
  assert cgroup_memory_available(root, proc) == _GIB
