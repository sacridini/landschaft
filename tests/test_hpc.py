# SPDX-License-Identifier: GPL-3.0-or-later
"""Batches in a cluster's batch jobs: the part of a job array, the record, SIGTERM and requeue, the
Slurm job script and the commands (no scheduler needed: its variables are simulated)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import landschaft as ls
from landschaft import _cli, _jobs

_JOB_VARIABLES = ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_MIN",
                  "SLURM_ARRAY_TASK_MAX", "SLURM_ARRAY_TASK_STEP", "SLURM_ARRAY_TASK_COUNT", "PBS_JOBID",
                  "PBS_ARRAY_INDEX", "PBS_ARRAYID", "LSB_JOBID", "LSB_JOBINDEX"]


@pytest.fixture(autouse=True)
def _no_job(monkeypatch):
    for v in _JOB_VARIABLES:
        monkeypatch.delenv(v, raising=False)


@pytest.fixture(scope="module")
def tiles(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiles")
    forest = ls.load_raster(ls.example_data("input"))
    for i in range(5):
        ls.write_raster(forest[i * 100:(i + 1) * 100, :200], folder / f"t{i}.tif")
    return folder


def _array(monkeypatch, task: int, first: int = 0, last: int = 2, step: int = 1, count: int | None = None):
    monkeypatch.setenv("SLURM_JOB_ID", f"77{task}")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "77")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task))
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", str(first))
    monkeypatch.setenv("SLURM_ARRAY_TASK_MAX", str(last))
    monkeypatch.setenv("SLURM_ARRAY_TASK_STEP", str(step))
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", str(count if count is not None else (last - first) // step + 1))


def test_parse_shard():
    assert _jobs.parse_shard("1/3")[:2] == (1, 3)
    assert _jobs.parse_shard((0, 2))[:2] == (0, 2)
    for bad in ("3/3", "-1/2", "1", "a/b", "0/0"):
        with pytest.raises(ValueError):
            _jobs.parse_shard(bad)
    part = _jobs.parse_shard("1/3")
    assert [i for i in range(8) if part.takes(i)] == [1, 4, 7]
    assert _jobs.ranges([3, 4, 5, 9, 11, 12]) == "3-5,9,11-12"


def test_array_split_is_kept(tmp_path, monkeypatch):
    # outside an array: every input
    assert _jobs.resolve_shard(tmp_path, "array") is None
    _array(monkeypatch, 4, first=2, last=6, step=2)          # --array=2-6:2: tasks 2, 4, 6
    part = _jobs.resolve_shard(tmp_path, "array")
    assert (part.index, part.count, part.source) == (1, 3, "Slurm job 77_4")
    assert json.loads((tmp_path / _jobs.STATE / _jobs.SPLIT).read_text()) == {"count": 3, "first": 2, "step": 2}
    # submitted again for the task that failed (--array=6): the same part of the same split
    _array(monkeypatch, 6, first=6, last=6)
    assert _jobs.resolve_shard(tmp_path, "array")[:2] == (2, 3)
    _array(monkeypatch, 5, first=5, last=5)
    with pytest.raises(ValueError, match="not one of the 3 tasks"):
        _jobs.resolve_shard(tmp_path, "array")
    # a first run that is not a range cannot tell the split
    _array(monkeypatch, 3, first=3, last=17, count=2)        # --array=3,17
    with pytest.raises(ValueError, match="not a range"):
        _jobs.resolve_shard(tmp_path / "other", "array")
    # an explicit part wins over the array's
    assert _jobs.resolve_shard(tmp_path / "other", "0/4")[:2] == (0, 4)


def test_pbs_and_lsf_arrays_ask_for_the_part(tmp_path, monkeypatch):
    monkeypatch.setenv("PBS_ARRAY_INDEX", "3")
    with pytest.raises(ValueError, match="PBS_ARRAY_INDEX"):
        _jobs.resolve_shard(tmp_path, "array")
    monkeypatch.delenv("PBS_ARRAY_INDEX")
    monkeypatch.setenv("LSB_JOBINDEX", "2")
    with pytest.raises(ValueError, match="LSB_JOBINDEX"):
        _jobs.resolve_shard(tmp_path, "array")


def test_run_names(monkeypatch):
    assert "-" in _jobs.run_name()                           # host-pid
    _array(monkeypatch, 3)
    assert _jobs.run_name() == "77_3"


def test_parts_cover_the_whole_batch(tiles, tmp_path):
    whole = ls.batch("accounting", tiles, tmp_path / "whole")
    assert not (tmp_path / "whole" / _jobs.STATE).exists()    # no record on a desktop without shard=
    parts = [ls.batch("accounting", tiles, tmp_path / "parts", shard=f"{i}/2") for i in range(2)]
    assert [r.input.name for r in parts[0]] == ["t0.tif", "t2.tif", "t4.tif"]
    assert [r.input.name for r in parts[1]] == ["t1.tif", "t3.tif"]
    for name in ("t0", "t3"):
        a = ls.load_raster(tmp_path / "whole" / f"{name}_accounting_ids.tif")
        b = ls.load_raster(tmp_path / "parts" / f"{name}_accounting_ids.tif")
        np.testing.assert_array_equal(a.values, b.values)
    assert len(whole) == 5
    status = ls.batch_status(tmp_path / "parts")
    assert status.ok and len(status.done) == 5 and status.tool == "accounting"
    assert "Every input is done." in str(status)
    # no partial files left
    assert not [p for p in (tmp_path / "parts").iterdir() if p.name.startswith(".") and p.is_file()]


def test_status_failed_missing_and_tasks(tiles, tmp_path, monkeypatch):
    out = tmp_path / "out"
    _array(monkeypatch, 1, first=0, last=1)

    def tool(data, n_threads=None):
        if float(data.mean()) > 1.4:
            raise ValueError("too much forest")
        return ls.accounting(data, n_threads=n_threads)

    tool.__name__ = "fussy"
    runs = ls.batch(tool, tiles, out, shard="array")
    assert [r.input.name for r in runs] == ["t1.tif", "t3.tif"]
    status = ls.batch_status(out)
    failed = [r.input.name for r in runs if not r.ok]
    assert sorted(status.failed) == failed and status.missing == ["t0.tif", "t2.tif", "t4.tif"]
    assert all(e["run"] == "77_1" and "too much forest" in e["error"] for e in status.failed.values())
    assert status.tasks(status.left) == ([0, 1] if failed else [0])
    text = status.describe()
    assert "sbatch --array=" in text and "not started: t0.tif, t2.tif, t4.tif" in text
    # the record of a run that did not end: its inputs in progress
    record = json.loads(next((out / _jobs.STATE / _jobs.RUNS).glob("*.json")).read_text())
    assert record["ended"] is not None and record["shard"] == [1, 2]


def test_resume_ignores_partial_outputs(tiles, tmp_path):
    out = tmp_path / "mspa"
    first = ls.mspa_batch(tiles, out, shard="0/5")
    dst = first[0].output
    # a run killed while writing leaves only a partial file under another name: not taken as done
    (out / f".{dst.stem}.123-456.part.tif").write_bytes(b"half")
    dst.unlink()
    again = ls.mspa_batch(tiles, out, shard="0/5")
    assert again[0].ok and again[0].seconds > 0 and dst.exists()
    assert ls.batch_status(out).done == ["t0.tif"]


def _stopping_tool(calls):
    def tool(data, n_threads=None):
        calls.append(1)
        if len(calls) == 2:  # SIGTERM while the second raster works (the handler the batch installed)
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return ls.accounting(data, n_threads=n_threads)

    tool.__name__ = "slowly"
    return tool


def test_sigterm_stops_and_requeues(tiles, tmp_path, monkeypatch):
    requeued = []
    monkeypatch.setattr(_jobs, "requeue", lambda: requeued.append(1) or "77_0")
    monkeypatch.setattr(_jobs, "time_left", lambda: 100.0)
    _array(monkeypatch, 0, first=0, last=0)
    calls: list = []
    with pytest.raises(_jobs.Stopped) as info:
        ls.batch(_stopping_tool(calls), tiles, tmp_path / "out", shard="array", requeue=True, n_workers=1)
    stopped = info.value
    assert len(calls) == 2 and requeued == [1] and stopped.requeued == "77_0"
    assert [r is not None for r in stopped.results] == [True, True, False, False, False]
    assert isinstance(stopped, KeyboardInterrupt)
    record = json.loads(next((tmp_path / "out" / _jobs.STATE / _jobs.RUNS).glob("*.json")).read_text())
    assert record["stopped"] == "SIGTERM" and record["requeued"] == "77_0" and len(record["done"]) == 2
    # the handler is put back
    assert signal.getsignal(signal.SIGTERM) is not None
    # the task requeued runs the rest
    rest = ls.batch(_stopping_tool([1, 1, 1]), tiles, tmp_path / "out", shard="array", requeue=True)
    assert all(r.ok for r in rest) and ls.batch_status(tmp_path / "out").ok


def test_no_requeue_without_progress_or_when_cancelled(tiles, tmp_path, monkeypatch):
    requeued = []
    monkeypatch.setattr(_jobs, "requeue", lambda: requeued.append(1) or "77_0")
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    # cancelled (scancel): an hour left
    monkeypatch.setattr(_jobs, "time_left", lambda: 3600.0)
    with pytest.raises(_jobs.Stopped) as info:
        ls.batch(_stopping_tool([]), tiles, tmp_path / "a", requeue=True, n_workers=1)
    assert requeued == [] and info.value.requeued is None
    # no input ended in this run
    monkeypatch.setattr(_jobs, "time_left", lambda: 60.0)
    stop = _jobs.Stopping(True, lambda: False, out=lambda text: None)
    stop.finish()
    assert requeued == []
    stop = _jobs.Stopping(True, lambda: True, out=lambda text: None)
    stop.finish()
    stop.finish()  # once
    assert requeued == [1] and stop.requeued == "77_0"


def test_time_left_parsing():
    assert _jobs._seconds("1-02:03:04") == 93784
    assert _jobs._seconds("02:03:04") == 7384
    assert _jobs._seconds("03:04") == 184
    assert _jobs._seconds("UNLIMITED") is None and _jobs._seconds("INVALID") is None


def test_memory_helpers():
    assert _jobs.slurm_bytes("32G") == 32 * 2**30 and _jobs.slurm_bytes("500M") == 500 * 2**20
    assert _jobs.slurm_bytes("4000") == 4000 * 2**20
    assert _jobs.batch_memory("32G") == "31744MiB"
    assert _jobs.suggest_memory(10**8, 32, 16) == "61G"      # 16 rasters of 1e8 pixels at 32 B, + 1 GB, + 25%
    assert _jobs.suggest_memory(1000, 32, 4) == "2G"
    with pytest.raises(ValueError):
        _jobs.slurm_bytes("lots")


def test_slurm_script(tiles, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    job = _jobs.write_script("fad", str(tiles), out, {"scales": [7, 13]}, cpus=2, time_limit="1:00:00",
                             partition="short", setup=["module load gdal"], sbatch=["--qos=long"])
    assert (job.inputs, job.tasks) == (5, 3)
    text = job.path.read_text()
    for line in ("#SBATCH --array=0-2", "#SBATCH --cpus-per-task=2", "#SBATCH --time=1:00:00",
                 "#SBATCH --partition=short", "#SBATCH --qos=long", "#SBATCH --signal=B:TERM@120",
                 "#SBATCH --requeue", "#SBATCH --mem=2G", "module load gdal"):
        assert line in text, line
    command = text.splitlines()[-1]
    assert command.startswith("exec ") and "-m landschaft batch fad" in command
    assert "--shard array --requeue" in command and "'scales=[7, 13]'" in command and "--max-memory" in command
    assert json.loads((out / _jobs.STATE / _jobs.SPLIT).read_text())["count"] == 3
    with pytest.raises(ValueError):
        _jobs.write_script("not_a_tool", str(tiles), out, {}, cpus=2)
    # the CLI prints it and how to submit it
    _cli.main(["slurm", "accounting", str(tiles), str(tmp_path / "cli"), "--cpus", "8", "--mem", "8G", "--tasks", "2"])
    printed = capsys.readouterr().out
    assert "#SBATCH --array=0-1" in printed and "#SBATCH --mem=8G" in printed and "sbatch " in printed


def test_cli_batch_and_status(tiles, tmp_path, capsys):
    assert _cli.parse_params(["edge_width=2", "name=forest", "scales=[3,5]", "transition=False", "x=none"]) == \
        {"edge_width": 2, "name": "forest", "scales": [3, 5], "transition": False, "x": "none"}
    with pytest.raises(ValueError):
        _cli.parse_params(["oops"])
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["batch", "mspa", str(tiles), str(out), "--shard", "0/2", "edge_width=2"])
    assert exit_info.value.code == 0
    assert "3 of 3 rasters done" in capsys.readouterr().out
    assert sorted(p.name for p in out.glob("*.tif")) == ["t0_8_2_1_1.tif", "t2_8_2_1_1.tif", "t4_8_2_1_1.tif"]
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["status", str(out)])
    assert exit_info.value.code == 3 and "2 not done" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["batch", "mspa", str(tiles), str(out), "edge_width=2"])
    assert exit_info.value.code == 0
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["status", str(out)])
    assert exit_info.value.code == 0
    with pytest.raises(SystemExit) as exit_info:
        _cli.main(["status", str(tmp_path / "nothing")])
    assert exit_info.value.code == 1


@pytest.mark.skipif(sys.platform == "win32", reason="a real SIGTERM to another process (POSIX)")
def test_real_sigterm(tiles, tmp_path):
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        import landschaft as ls

        def slow(data, n_threads=None):
            print("started", flush=True)
            time.sleep(1.5)
            return ls.accounting(data, n_threads=n_threads)

        try:
            ls.batch(slow, {str(tiles)!r}, {str(tmp_path / 'out')!r}, n_workers=1, shard="0/1")
        except ls.batch_module.Stopped as e:
            print("stopped", sum(r is not None for r in e.results), flush=True)
            sys.exit(143)
    """).replace("ls.batch_module.Stopped", "__import__('landschaft.batch', fromlist=['Stopped']).Stopped"))
    proc = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True,
                            env={**os.environ, "LANDSCHAFT_NUM_THREADS": "1"})
    assert proc.stdout.readline().strip() == "started"
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 143 and out.strip().endswith("stopped 1")
    assert len(ls.batch_status(tmp_path / "out").done) == 1
