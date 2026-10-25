# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``landschaft`` command: lend this computer to a group of helpers and manage the group; batches
of many rasters, in a cluster's job array too.

    landschaft helpers invite         # on the computer that will use helpers: shows a code
    landschaft worker --join CODE     # on each helper, the first time
    landschaft worker                 # afterwards: wait for work until Ctrl+C
    landschaft batch fad tiles/ out/  # a tool on every raster of a folder
    landschaft slurm fad tiles/ out/ --cpus 16   # the same, split among the tasks of a Slurm job array
    landschaft status out/            # what is done, failed and left
    landschaft fetch                  # download beforehand what landschaft downloads on first use
"""

from __future__ import annotations

import argparse
import sys

from ._dist import _group


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="landschaft", description="landschaft: lend this computer to the "
                                     "group of helpers or manage the group, run a tool on many rasters (also as a "
                                     "Slurm job array), and see what landschaft finds on this computer.")
    parser.add_argument("--version", action="store_true", help="Show landschaft's version")
    sub = parser.add_subparsers(dest="command")

    wk = sub.add_parser("worker", help="Lend this computer to the group: wait for tasks and run them "
                                       "(see `landschaft helpers invite`)")
    wk.add_argument("--join", default=None, metavar="CODE", help="Join a group with the code shown by "
                    "`landschaft helpers invite` on one of its computers (the first time only)")
    wk.add_argument("--cores", type=int, default=None, help="Cores to lend (default: all but one)")
    wk.add_argument("--connect", default=None, metavar="HOST[:PORT]", help="The coordinator's address, when the "
                    "broadcasts of the local network do not reach it")
    wk.add_argument("--map", action="append", default=None, metavar="SRC=DST", help="A shared folder mounted "
                    "elsewhere here: files the coordinator sees under SRC are read under DST (repeatable), "
                    "e.g. --map Z:/=/mnt/nas")
    wk.add_argument("--cache-dir", default=None, help="Where received input files are kept (default: the user's "
                    "cache folder)")
    wk.add_argument("--cache-gb", type=float, default=20.0, help="Space the kept files may take, in GB (default: 20)")

    hp = sub.add_parser("helpers", help="The group of computers that work together")
    hp_sub = hp.add_subparsers(dest="helpers_command")
    inv = hp_sub.add_parser("invite", help="Show a code with which other computers join this one's group")
    inv.add_argument("--minutes", type=float, default=10.0, help="How long the code is valid (default: 10)")
    inv.add_argument("--port", type=int, default=_group.TCP_PORT, help=f"TCP port (default: {_group.TCP_PORT})")
    ls_ = hp_sub.add_parser("list", help="The helpers of the group waiting for work on the local network")
    ls_.add_argument("--seconds", type=float, default=5.0, help="How long to listen (default: 5)")
    fw = hp_sub.add_parser("firewall", help="Let helpers reach this computer (Windows: creates the firewall rules)")
    fw.add_argument("--port", type=int, default=_group.TCP_PORT, help=f"TCP port (default: {_group.TCP_PORT})")
    hp_sub.add_parser("leave", help="Take this computer out of its group")

    bt = sub.add_parser("batch", help="Run a tool on many rasters and write its outputs (ls.batch); in a job array, "
                                      "with --shard array, the task's part of them")
    _batch_arguments(bt)
    bt.add_argument("--shard", default=None, metavar="I/N|array", help="Only part I (from 0) of N of the rasters; "
                    "'array': this Slurm array task's part (all of them outside an array)")
    bt.add_argument("--requeue", action="store_true", help="In a Slurm job stopped near its time limit (SIGTERM), "
                    "requeue it once the rasters at work end, to continue")
    bt.add_argument("--workers", type=int, default=None, help="Rasters at once (default: the threads / "
                    "--threads-per-worker)")
    bt.add_argument("--max-memory", default=None, help="Start a raster only while the estimates fit, e.g. 16GB")
    bt.add_argument("--overwrite", action="store_true", help="Compute again the rasters already done")

    st = sub.add_parser("status", help="What the batches of an output folder did: done, failed, left, and the "
                                       "job array tasks to submit again")
    st.add_argument("output_dir")

    sl = sub.add_parser("slurm", help="Write the Slurm job script that splits a batch among the tasks of a job "
                                      "array (and submit it with --submit)")
    _batch_arguments(sl)
    sl.add_argument("--cpus", type=int, default=16, help="Processors of each task, --cpus-per-task (default: 16)")
    sl.add_argument("--mem", default=None, help="Memory of each task, as Slurm takes it (32G); default: suggested "
                    "from the largest raster and the tool")
    sl.add_argument("--time", default=None, help="Time limit of each task (2:00:00, 1-00:00:00); default: the "
                    "partition's")
    sl.add_argument("--partition", default=None, help="The partition (queue)")
    sl.add_argument("--account", default=None, help="The account to charge")
    sl.add_argument("--tasks", type=int, default=None, help="Tasks of the array (default: the rasters / the rasters "
                    "a task runs at once, at most 1000)")
    sl.add_argument("--max-running", type=int, default=None, help="Tasks running at once (--array=0-49%%N)")
    sl.add_argument("--sbatch", action="append", default=[], metavar="OPTION", help="Another #SBATCH line, e.g. "
                    "--sbatch=--qos=long (repeatable)")
    sl.add_argument("--setup", action="append", default=[], metavar="LINE", help="A line run before landschaft, "
                    "e.g. --setup 'module load gdal' (repeatable)")
    sl.add_argument("--script", default=None, help="Where to write the script (default: "
                    "<output_dir>/.landschaft-batch/<tool>.sbatch)")
    sl.add_argument("--submit", action="store_true", help="Submit it with sbatch")

    ft = sub.add_parser("fetch", help="Download beforehand the data landschaft downloads on first use (on a "
                                      "cluster's login node, for compute nodes without internet access)")
    ft.add_argument("what", nargs="?", default="all", choices=("all", "sdm", "examples"),
                    help="sdm: the reference data of ls.sdm (~40 MB); examples: the example data that is "
                         "downloaded; all (default): both")
    parser.landschaft_commands = {"worker": wk, "helpers": hp}
    return parser


def _batch_arguments(p) -> None:
    p.add_argument("tool", help="A tool of ls.BATCH_TOOLS: mspa, fad, accounting, spa, landscape_metrics, ...")
    p.add_argument("inputs", help="A folder of rasters, or a quoted glob ('tiles/*.tif')")
    p.add_argument("output_dir", help="Where the outputs go")
    p.add_argument("params", nargs="*", metavar="KEY=VALUE", help="Parameters of the tool, e.g. edge_width=2 "
                   "variant=6 scales=[3,5,7] (numbers, true/false, lists in JSON; anything else is text)")
    p.add_argument("--threads-per-worker", type=int, default=1, help="Threads inside each raster (default: 1)")
    p.add_argument("--table-format", default="csv", choices=("csv", "parquet"), help="Format of the tables")


def parse_params(items) -> dict:
    """``["edge_width=2", "name=forest", "scales=[3,5]"]`` -> the tool's keyword arguments."""
    import json

    params = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.isidentifier():
            raise ValueError(f"parameters are KEY=VALUE, got {item!r}")
        text = {"True": "true", "False": "false", "None": "null"}.get(value, value)
        try:
            params[key] = json.loads(text)
        except ValueError:
            params[key] = value
    return params


def main(argv: list | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(f"landschaft {_group.version()}")
    elif args.command == "worker":
        from ._dist._worker import run_worker

        try:
            run_worker(cores=args.cores, connect=args.connect, join=args.join, maps=args.map or (),
                       cache_dir=args.cache_dir, cache_gb=args.cache_gb)
        except (ValueError, OSError, RuntimeError) as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.command == "helpers":
        _helpers(args, parser)
    elif args.command == "batch":
        _batch(args)
    elif args.command == "status":
        from ._jobs import batch_status

        try:
            status = batch_status(args.output_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(status.describe())
        sys.exit(0 if status.ok else 3)
    elif args.command == "slurm":
        _slurm(args)
    elif args.command == "fetch":
        from ._download import prefetch

        try:
            prefetch(args.what)
        except OSError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


def _batch(args) -> None:
    from .batch import batch
    from ._jobs import Stopped

    try:
        params = parse_params(args.params)
        results = batch(args.tool, args.inputs, args.output_dir, n_workers=args.workers,
                           threads_per_worker=args.threads_per_worker, max_memory=args.max_memory,
                           overwrite=args.overwrite, table_format=args.table_format, shard=args.shard,
                           requeue=args.requeue, **params)
    except Stopped as e:
        ended = sum(r is not None for r in e.results)
        print(f"Stopped by {e.signal_name}: {ended} of {len(e.results)} rasters ended"
              + (f"; requeued {e.requeued}" if e.requeued else ""))
        sys.exit(143)
    except (ValueError, OSError, RuntimeError, TypeError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    failed = [r for r in results if not r.ok]
    print(f"{args.tool}: {len(results) - len(failed)} of {len(results)} rasters done in {args.output_dir}")
    for r in failed:
        print(f"  failed: {r.input.name}: {r.error}")
    sys.exit(1 if failed else 0)


def _slurm(args) -> None:
    from . import _jobs

    try:
        job = _jobs.write_script(args.tool, args.inputs, args.output_dir, parse_params(args.params), cpus=args.cpus,
                                 threads_per_worker=args.threads_per_worker, mem=args.mem, time_limit=args.time,
                                 partition=args.partition, account=args.account, tasks=args.tasks,
                                 max_running=args.max_running, sbatch=args.sbatch, setup=args.setup,
                                 script=args.script, table_format=args.table_format)
    except (ValueError, OSError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(job.text)
    print(f"Written: {job.path} ({job.inputs} rasters, {job.tasks} tasks)")
    if args.submit:
        try:
            print(_jobs.submit(job.path))
        except RuntimeError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        print(f"Submit it with: sbatch {job.path}")


def _helpers(args, parser) -> None:
    from . import _dist

    if args.helpers_command == "invite":
        try:
            joined = _dist.invite_helpers(args.minutes, port=args.port)
        except (OSError, RuntimeError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"{len(joined)} computer(s) joined." if joined else "No computer joined.")
    elif args.helpers_command == "list":
        try:
            found = _dist.list_helpers(args.seconds)
        except (OSError, RuntimeError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        if not found:
            print("No helper is waiting for work (busy helpers do not show).")
        for h in found:
            print(f"{h['name']}  {h['address']}  {h['cores']} cores  landschaft {h['version']}")
    elif args.helpers_command == "firewall":
        from ._dist._system import open_firewall

        print(open_firewall(args.port, _group.UDP_PORT))
    elif args.helpers_command == "leave":
        print("This computer left its group." if _group.forget_secret() else "This computer is not in a group.")
    else:
        parser.landschaft_commands["helpers"].print_help()


if __name__ == "__main__":
    main()
