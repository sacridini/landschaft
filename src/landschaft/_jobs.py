# SPDX-License-Identifier: GPL-3.0-or-later
"""Batches inside a cluster's batch jobs (Slurm, PBS, LSF): the part of the inputs a task of a job
array processes, the record of what each run did, the time limit, and the Slurm job script.

The inputs, in their order, go to the parts in turn: input ``i`` to part ``i mod n``. A static
split needs no lock in the file system (NFS and Lustre differ on ``O_EXCL``). In a Slurm job
array the part is the task's, from ``SLURM_ARRAY_TASK_ID``; the split (how many parts, the first
task id, the step) is kept in the output folder the first time (``.landschaft-batch/split.json``),
so that the tasks submitted again later, ``--array=3,17`` for those that failed, take the same
inputs as before. ``"i/n"`` gives the part by hand, for other schedulers.

The record of an output folder (``.landschaft-batch/``):

- ``inputs.json``: the tool, its parameters, every input in the order of the split and the file
  that marks it done;
- ``split.json``: how a job array splits them;
- ``runs/<run>.json``: one per run (a task of an array, a process): its inputs, those it
  finished, those that failed and why, and whether it was stopped.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Sequence

from ._threads import JOB_VARIABLES

STATE = ".landschaft-batch"
SPLIT = "split.json"
MANIFEST = "inputs.json"
RUNS = "runs"
MAX_ARRAY = 1000          # Slurm's default MaxArraySize is 1001: ids 0 to 1000
SIGNAL_SECONDS = 120      # SIGTERM this long before the time limit (#SBATCH --signal=B:TERM@120)
GRACE_SECONDS = 90        # after SIGTERM, how long the inputs at work may take before the task requeues itself
NEAR_LIMIT_SECONDS = 600  # a SIGTERM with more time left than this is a cancellation: no requeue


class Stopped(KeyboardInterrupt):
    """A batch stopped by SIGTERM (a scheduler's warning before the job's time limit, or its
    cancellation): no input started after it; ``results`` holds those that ended, in input order
    (``None`` for the inputs not processed)."""

    def __init__(self, signal_name: str = "SIGTERM", results: list | None = None, requeued: str | None = None):
        super().__init__(signal_name)
        self.signal_name = signal_name
        self.results = results or []
        self.requeued = requeued


def in_job() -> str | None:
    """The scheduler of the batch job this process runs in (Slurm, PBS, LSF), None outside one."""
    return next((name for name, variable in JOB_VARIABLES if os.environ.get(variable, "").strip()), None)


# ---- the split ----------------------------------------------------------------------------------

class ArrayTask(NamedTuple):
    """The task of a job array this process runs in."""

    scheduler: str
    job: str            # the array's job id
    task: int           # this task's id
    first: int          # the ids of the array: first, last, step, count
    last: int
    step: int
    count: int


class Shard(NamedTuple):
    """Part ``index`` of ``count`` (from 0)."""

    index: int
    count: int
    source: str         # "shard=" or the array's "Slurm job 123_4"
    first: int = 0      # the array task id of part 0, and between parts
    step: int = 1

    def takes(self, position: int) -> bool:
        return position % self.count == self.index


def array_task() -> ArrayTask | None:
    """The job array task of this process (Slurm), None outside one. PBS and LSF arrays do not tell
    how many tasks they have: ValueError, asking for ``shard="i/n"``."""
    env = os.environ
    if env.get("SLURM_ARRAY_TASK_ID"):
        try:
            task = int(env["SLURM_ARRAY_TASK_ID"])
            first, last = int(env["SLURM_ARRAY_TASK_MIN"]), int(env["SLURM_ARRAY_TASK_MAX"])
            step = int(env.get("SLURM_ARRAY_TASK_STEP") or 1)
            count = int(env.get("SLURM_ARRAY_TASK_COUNT") or (last - first) // max(step, 1) + 1)
        except (KeyError, ValueError) as e:
            raise ValueError(f"the Slurm array variables are incomplete ({e}); give the part as shard='i/n'") from None
        return ArrayTask("Slurm", env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID", "?"), task, first, last,
                         max(step, 1), count)
    for name, variable in (("PBS", "PBS_ARRAY_INDEX"), ("PBS", "PBS_ARRAYID"), ("LSF", "LSB_JOBINDEX")):
        index = env.get(variable)
        if index and index != "0":
            raise ValueError(f"a {name} job array does not tell its tasks how many there are: give the part as "
                             f"shard='i/n' (--shard ${variable}/<number of tasks> with indices from 0; from 1, "
                             f"--shard $(({variable} - 1))/<number of tasks>)")
    return None


def parse_shard(shard) -> Shard:
    """``"i/n"`` or ``(i, n)``: part ``i`` (from 0) of ``n``."""
    try:
        index, count = (int(v) for v in (str(shard).split("/") if isinstance(shard, str) else shard))
    except (TypeError, ValueError):
        raise ValueError(f"shard takes 'i/n' (part i, from 0, of n) or 'array', got {shard!r}") from None
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard {index}/{count}: i goes from 0 to n - 1")
    return Shard(index, count, "shard=")


def read_split(output_dir) -> dict | None:
    try:
        split = json.loads((Path(output_dir) / STATE / SPLIT).read_text())
        return split if all(isinstance(split.get(k), int) for k in ("count", "first", "step")) else None
    except (OSError, ValueError, AttributeError):
        return None


def write_split(output_dir, count: int, first: int = 0, step: int = 1, **more) -> None:
    """Keep the split of the output folder (tasks starting together write the same)."""
    write_json(Path(output_dir) / STATE / SPLIT, {"count": count, "first": first, "step": step, **more})


def resolve_shard(output_dir, shard) -> Shard | None:
    """The part of the inputs this process runs: ``shard="i/n"`` (or ``(i, n)``); ``"array"``: this job
    array task's (with the split the output folder keeps), all of them outside an array; None: all."""
    if shard is None:
        return None
    if not (isinstance(shard, str) and shard.strip().lower() == "array"):
        return parse_shard(shard)
    task = array_task()
    if task is None:
        return None
    split = read_split(output_dir)
    if split is None:
        if task.count != (task.last - task.first) // task.step + 1:
            raise ValueError(f"this Slurm array ({task.count} tasks between {task.first} and {task.last}) is not "
                             f"a range, and {output_dir} keeps no split of an earlier run: submit a range "
                             "(--array=0-49), or write the script with `landschaft slurm`")
        split = {"count": task.count, "first": task.first, "step": task.step}
        write_split(output_dir, **split)
    offset = task.task - split["first"]
    if offset < 0 or offset % split["step"] or offset // split["step"] >= split["count"]:
        last = split["first"] + (split["count"] - 1) * split["step"]
        raise ValueError(f"array task {task.task} is not one of the {split['count']} tasks that split the inputs of "
                         f"{output_dir} (ids {split['first']} to {last}"
                         f"{', step ' + str(split['step']) if split['step'] != 1 else ''}, kept in {STATE}/{SPLIT} "
                         "since its first run; delete it to split them otherwise)")
    return Shard(offset // split["step"], split["count"], f"{task.scheduler} job {task.job}_{task.task}",
                 split["first"], split["step"])


def run_name() -> str:
    """This run's name among the runs writing into the same output folder: the job and array task
    (``123456_7``, or the job id), else the computer and process (``node12-4242``)."""
    env = os.environ
    job = env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID") or env.get("PBS_JOBID") or env.get("LSB_JOBID")
    if job:
        task = env.get("SLURM_ARRAY_TASK_ID") or env.get("PBS_ARRAY_INDEX") or env.get("LSB_JOBINDEX")
        name = f"{job}_{task}" if task else job
    else:
        name = f"{socket.gethostname()}-{os.getpid()}"
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


# ---- the record ---------------------------------------------------------------------------------

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, data: dict) -> None:
    """Atomically: a reader never sees half a file, and runs writing the same file at once leave one
    whole copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}")
    part.write_text(json.dumps(data, indent=1))
    os.replace(part, path)


def write_manifest(output_dir, tool: str, params: dict, inputs: Sequence[Path], markers: Sequence[str]) -> None:
    """Every input of the batch, in the order the split follows, with the file that marks it done."""
    write_json(Path(output_dir) / STATE / MANIFEST, {"tool": tool, "params": params,
                                                     "inputs": [str(p) for p in inputs], "markers": list(markers)})


class Journal:
    """The record of one run, rewritten as inputs finish (best effort: it never stops the batch)."""

    def __init__(self, output_dir, tool: str, shard: Shard | None, inputs: Sequence[Path], skipped: Sequence[Path]):
        self.path = Path(output_dir) / STATE / RUNS / f"{run_name()}.json"
        self._lock = threading.Lock()
        self.new = 0
        self.record = {"run": run_name(), "host": socket.gethostname(), "pid": os.getpid(), "started": _now(),
                       "tool": tool, "shard": [shard.index, shard.count] if shard is not None else None,
                       "inputs": [p.name for p in inputs], "skipped": [p.name for p in skipped],
                       "done": [], "failed": {}, "stopped": None, "requeued": None, "ended": None}
        self._save()

    def _save(self) -> None:
        try:
            write_json(self.path, self.record)
        except OSError:
            pass

    def result(self, src: Path, error: str | None, skipped: bool = False) -> None:
        with self._lock:
            if error:
                self.record["failed"][src.name] = {"error": str(error).strip()[-4000:], "at": _now()}
            else:
                self.record["failed"].pop(src.name, None)
                if not skipped:
                    self.record["done"].append(src.name)
                    self.new += 1
            self._save()

    def end(self, stopped: str | None = None, requeued: str | None = None) -> None:
        with self._lock:
            self.record.update(stopped=stopped, requeued=requeued, ended=_now())
            self._save()


@dataclass
class BatchStatus:
    """The inputs of an output folder, across every run of :func:`landschaft.batch` that wrote into it.

    Attributes
    ----------
    output_dir
        The folder.
    tool
        The tool of the batch.
    inputs
        Every input (file names), in the order of the split.
    done, failed, missing
        Inputs finished, failed (name -> ``{"error", "run"}``) and not done.
    running
        Inputs of runs that have not ended (still running, or killed): name -> run.
    split
        How a job array splits the inputs, or ``None``.
    """

    output_dir: Path
    tool: str
    inputs: list[str]
    done: list[str] = field(default_factory=list)
    failed: dict[str, dict] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    running: dict[str, str] = field(default_factory=dict)
    split: dict | None = None

    @property
    def left(self) -> list[str]:
        """The inputs failed or not done."""
        return list(self.failed) + self.missing

    @property
    def ok(self) -> bool:
        return not self.left

    def tasks(self, names: Sequence[str]) -> list[int]:
        """The array task ids that process ``names`` (with the split the folder keeps)."""
        if not self.split:
            return []
        position = {name: i for i, name in enumerate(self.inputs)}
        count, first, step = self.split["count"], self.split["first"], self.split["step"]
        return sorted({first + (position[n] % count) * step for n in names if n in position})

    def describe(self, shown: int = 10) -> str:
        """What ``landschaft status`` prints."""
        lines = [f"{self.tool} in {self.output_dir}: {len(self.inputs)} inputs, {len(self.done)} done, "
                 f"{len(self.failed)} failed, {len(self.missing)} not done"
                 + (f" ({len(self.running)} of them in runs that have not ended)" if self.running else "")]
        for name, entry in list(self.failed.items())[:shown]:
            error = (entry.get("error") or "").strip().splitlines()
            lines.append(f"  failed: {name} (run {entry.get('run')}): {error[-1] if error else 'no message'}")
        if len(self.failed) > shown:
            lines.append(f"  ... and {len(self.failed) - shown} more failed")
        waiting = [n for n in self.missing if n not in self.running]
        if waiting:
            more = f" ... and {len(waiting) - shown} more" if len(waiting) > shown else ""
            lines.append(f"  not started: {', '.join(waiting[:shown])}{more}")
        if self.running:
            names = sorted(self.running)
            more = f" ... and {len(names) - shown} more" if len(names) > shown else ""
            lines.append(f"  in runs that have not ended (still running, or killed): {', '.join(names[:shown])}{more}")
        left = self.left
        if not left:
            lines.append("Every input is done.")
        elif self.split:
            script = self.split.get("script") or "<the job script>"
            lines.append(f"To process the {len(left)} left: sbatch --array={ranges(self.tasks(left))} {script}")
        else:
            lines.append(f"To process the {len(left)} left: run the batch again (without shard= for all of them; "
                         "finished inputs are kept).")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.describe()


def ranges(ids: Sequence[int]) -> str:
    """``[3, 4, 5, 9]`` -> ``"3-5,9"``, as Slurm's --array takes them."""
    out: list[list[int]] = []
    for i in sorted(set(ids)):
        if out and out[-1][1] + 1 == i:
            out[-1][1] = i
        else:
            out.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def batch_status(output_dir) -> BatchStatus:
    """What the runs of :func:`landschaft.batch` / :func:`landschaft.mspa_batch` did in an output folder:
    the inputs done, failed (with their error) and not done, and the job array tasks to submit again.

    The record is kept in ``<output_dir>/.landschaft-batch/`` by the batches that run in a batch job
    (Slurm, PBS, LSF) or with ``shard=`` or ``requeue=``. Also ``landschaft status <output_dir>`` on
    the command line.

    Parameters
    ----------
    output_dir
        The output folder of the batch.

    Returns
    -------
    BatchStatus
        ``print()`` it for a summary; ``.left`` lists the inputs still to do.

    Examples
    --------
    >>> from pathlib import Path
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))
    >>> Path("tiles").mkdir(exist_ok=True)
    >>> for row in (0, 500):
    ...     _ = ls.write_raster(forest[row:row + 500, :500], f"tiles/forest_{row}.tif")
    >>> _ = ls.batch("accounting", "tiles", "parts_out", shard="0/2")    # the first of two parts
    >>> status = ls.batch_status("parts_out")
    >>> status.done, status.missing
    (['forest_0.tif'], ['forest_500.tif'])
    """
    output_dir = Path(output_dir)
    try:
        manifest = json.loads((output_dir / STATE / MANIFEST).read_text())
    except (OSError, ValueError):
        raise FileNotFoundError(f"{output_dir} has no record of a batch ({STATE}/{MANIFEST}): it is written by "
                                "batches run in a batch job or with shard= or requeue=") from None
    names = [Path(p).name for p in manifest.get("inputs", [])]
    status = BatchStatus(output_dir, manifest.get("tool", "?"), names, split=read_split(output_dir))
    records = []
    for path in sorted((output_dir / STATE / RUNS).glob("*.json")):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    records.sort(key=lambda r: str(r.get("started", "")))
    failures: dict[str, tuple[str, dict]] = {}
    for r in records:
        for name, entry in (r.get("failed") or {}).items():
            failures[name] = (str(r.get("run")), entry)
        for name in r.get("done") or []:
            failures.pop(name, None)
        if r.get("ended") is None:
            for name in r.get("inputs") or []:
                if name not in (r.get("done") or []) and name not in (r.get("failed") or {}):
                    status.running[name] = str(r.get("run"))
    for name, marker in zip(names, manifest.get("markers", [])):
        if (output_dir / marker).exists():
            status.done.append(name)
            status.running.pop(name, None)
        elif name in failures:
            run, entry = failures[name]
            status.failed[name] = {**entry, "run": run}
            status.running.pop(name, None)
        else:
            status.missing.append(name)
    return status


# ---- the time limit -----------------------------------------------------------------------------

def _seconds(text: str) -> float | None:
    """Slurm's time left ("1-02:03:04", "02:03:04", "03:04") in seconds; None for UNLIMITED, INVALID."""
    text, days = text.strip(), 0
    try:
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
        parts = [int(v) for v in text.split(":")]
    except ValueError:
        return None
    if not 1 <= len(parts) <= 3:
        return None
    hours, minutes, seconds = [0] * (3 - len(parts)) + parts
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def time_left() -> float | None:
    """Seconds left to this Slurm job (``squeue %L``), None when it cannot tell."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job or shutil.which("squeue") is None:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return _seconds(out.stdout.splitlines()[0]) if out.returncode == 0 and out.stdout.strip() else None


def requeue() -> str | None:
    """Requeue this Slurm job (or array task): the id requeued, or None outside a Slurm job.
    RuntimeError when scontrol refuses."""
    env = os.environ
    if env.get("SLURM_ARRAY_JOB_ID") and env.get("SLURM_ARRAY_TASK_ID"):
        job = f"{env['SLURM_ARRAY_JOB_ID']}_{env['SLURM_ARRAY_TASK_ID']}"
    elif env.get("SLURM_JOB_ID"):
        job = env["SLURM_JOB_ID"]
    else:
        return None
    done = subprocess.run(["scontrol", "requeue", job], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f"scontrol requeue {job} failed: {(done.stderr or done.stdout).strip()}")
    return job


class Stopping:
    """SIGTERM as a stop: ``event`` is set, no input starts after it, and with ``requeue`` the Slurm
    task requeues itself once the inputs at work end or :data:`GRACE_SECONDS` pass, if this run
    finished at least one input (``progress()``) and the job is near its time limit (a cancellation
    by ``scancel`` leaves more time and is not requeued)."""

    def __init__(self, requeue: bool, progress, out=None):
        self.event = threading.Event()
        self.signal_name: str | None = None
        self.requeued: str | None = None
        self._requeue = requeue
        self._progress = progress
        self._out = out or (lambda text: print(text, file=sys.stderr, flush=True))
        self._done = threading.Lock()
        self._finished = False

    def _stop(self, signum, frame) -> None:
        if self.event.is_set():
            return
        self.signal_name = signal.Signals(signum).name
        self.event.set()
        self._out(f"landschaft: {self.signal_name}: no input starts after this; the ones at work may end")
        if self._requeue:
            threading.Thread(target=self._watch, name="landschaft-requeue", daemon=True).start()

    def _watch(self) -> None:
        time.sleep(GRACE_SECONDS)
        self.finish()

    def finish(self) -> None:
        """Requeue (once) if asked, with progress, near the time limit."""
        with self._done:
            if self._finished or not self._requeue:
                return
            self._finished = True
        if not self._progress():
            self._out("landschaft: not requeued: no input ended in this run (an input longer than the time limit "
                      "would come back forever); give the job more time")
            return
        left = time_left()
        if left is not None and left > NEAR_LIMIT_SECONDS:
            self._out(f"landschaft: not requeued: {left:.0f} s were left to the job (cancelled, not at its limit)")
            return
        try:
            self.requeued = requeue()
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            self._out(f"landschaft: could not requeue: {e}")
            return
        if self.requeued:
            self._out(f"landschaft: requeued {self.requeued}")


@contextmanager
def stopping(active: bool, requeue: bool, progress):
    """A :class:`Stopping` handling SIGTERM within the block (in the main thread, when ``active``)."""
    stop = Stopping(requeue, progress)
    if not active or threading.current_thread() is not threading.main_thread() or not hasattr(signal, "SIGTERM"):
        yield stop
        return
    previous = signal.signal(signal.SIGTERM, stop._stop)
    try:
        yield stop
    finally:
        signal.signal(signal.SIGTERM, previous)


# ---- the job script -----------------------------------------------------------------------------

class JobScript(NamedTuple):
    path: Path
    text: str
    inputs: int
    tasks: int


def slurm_bytes(mem: str) -> int:
    """Slurm's ``--mem`` (``32G``, ``500M``, ``4000``: megabytes) in bytes (Slurm's units are of 1024)."""
    text = str(mem).strip().upper().removesuffix("B")
    unit = text[-1] if text and text[-1] in "KMGT" else "M"
    try:
        number = float(text.rstrip("KMGT"))
    except ValueError:
        raise ValueError(f"--mem takes a size as Slurm does (32G, 500M), got {mem!r}") from None
    return int(number * 1024 ** "KMGT".index(unit) * 1024)


def suggest_memory(largest_pixels: int, bytes_per_pixel: int, workers: int) -> str:
    """``--mem`` of a task: the inputs at work at once, each as large as the largest, with the process's
    own (1 GB) and a quarter more, in whole GB (at least 2)."""
    rasters = workers * largest_pixels * bytes_per_pixel / 2**30
    return f"{max(2, math.ceil(1.25 * (1.0 + rasters)))}G"


def batch_memory(mem: str) -> str:
    """The ``max_memory`` of the batch of a task of ``--mem``: what the process's own 1 GB leaves."""
    return f"{max(2**30, slurm_bytes(mem) - 2**30) // 2**20}MiB"


def write_script(tool: str, inputs: str, output_dir, params: dict, *, cpus: int, threads_per_worker: int = 1,
                 mem: str | None = None, time_limit: str | None = None, partition: str | None = None,
                 account: str | None = None, tasks: int | None = None, max_running: int | None = None,
                 sbatch: Sequence[str] = (), setup: Sequence[str] = (), script: str | None = None,
                 table_format: str = "csv") -> JobScript:
    """Keep the split in the output folder and write the job script of a batch over a job array."""
    from .batch import BYTES_PER_PIXEL, _TOOL_BYTES_PER_PIXEL, _header_pixels, _resolve_tool
    from .collection import expand_sources

    name, _ = _resolve_tool(tool)
    if cpus < 1:
        raise ValueError("--cpus must be at least 1")
    paths = expand_sources(inputs)
    n = len(paths)
    if n == 0:
        raise ValueError(f"no rasters in {inputs}")
    workers = max(1, cpus // max(1, threads_per_worker))
    tasks = min(n, MAX_ARRAY, tasks if tasks else math.ceil(n / workers))
    if tasks < 1:
        raise ValueError("--tasks must be at least 1")
    output_dir = Path(output_dir).resolve()
    path = Path(script).resolve() if script else output_dir / STATE / f"{name}.sbatch"
    logs = output_dir / STATE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    old = read_split(output_dir)
    if old is not None and (old["count"], old["first"], old["step"]) != (tasks, 0, 1):
        print(f"! {output_dir} was split among {old['count']} tasks before; now among {tasks}. Tasks of the earlier "
              "split still running would take other inputs than this script's.", file=sys.stderr)
    write_split(output_dir, tasks, 0, 1, script=str(path))
    per_task = math.ceil(n / tasks)
    note = []
    if not mem:
        largest = max(_header_pixels(p) for p in paths)
        per_pixel = _TOOL_BYTES_PER_PIXEL.get(name, BYTES_PER_PIXEL)
        at_once = min(workers, per_task)
        mem = suggest_memory(largest, per_pixel, at_once)
        note = [f"# --mem: {at_once} rasters at once of up to {largest:,} pixels, at about {per_pixel} bytes per "
                f"pixel for {name}, plus 1 GB for the process, and a quarter more."]
    max_memory = batch_memory(mem)
    note.append(f"# The batch starts a raster only while its estimate fits in {max_memory} (--max-memory).")
    source = os.path.abspath(inputs) if os.path.exists(inputs) else inputs
    line = [sys.executable, "-m", "landschaft", "batch", name, source, str(output_dir), "--shard", "array",
            "--requeue", "--threads-per-worker", str(threads_per_worker), "--max-memory", max_memory]
    if table_format != "csv":
        line += ["--table-format", table_format]
    line += [f"{k}={json.dumps(v)}" for k, v in params.items()]
    array = f"0-{tasks - 1}" + (f"%{max_running}" if max_running else "")
    head = [
        "#!/bin/bash",
        f"# Written by `landschaft slurm`: {name} on {n} rasters of {source}, "
        + ("one per task of the array." if per_task == 1 else f"{per_task} or fewer per task of the array."),
        f"#   submit it:    sbatch {path}",
        f"#   follow it:    landschaft status {output_dir}",
        "# Each task takes its rasters from SLURM_ARRAY_TASK_ID and every processor it is given; finished rasters",
        "# are kept, and a task stopped at its time limit requeues itself to continue.",
        *note,
        f"#SBATCH --job-name=landschaft-{name}",
        f"#SBATCH --array={array}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
    ]
    if time_limit:
        head.append(f"#SBATCH --time={time_limit}")
    if partition:
        head.append(f"#SBATCH --partition={partition}")
    if account:
        head.append(f"#SBATCH --account={account}")
    head += [
        f"#SBATCH --output={logs / 'slurm-%A_%a.out'}",
        f"#SBATCH --signal=B:TERM@{SIGNAL_SECONDS}",
        "#SBATCH --requeue",
    ]
    head += [f"#SBATCH {extra}" for extra in sbatch]
    body = [""] + list(setup) + ([""] if setup else []) + ["exec " + " ".join(shlex.quote(str(a)) for a in line)]
    text = "\n".join(head + body) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, newline="\n")
    return JobScript(path, text, n, tasks)


def submit(path: Path) -> str:
    """``sbatch <path>``: what it answered (``Submitted batch job 123456``)."""
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is not here: run `landschaft slurm --submit` on the cluster's login node, or "
                           f"submit the script there with: sbatch {path}")
    done = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"sbatch failed: {(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()
