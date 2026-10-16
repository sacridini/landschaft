# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``landschaft`` command: lend this computer to a group of helpers, and manage the group.

    landschaft helpers invite         # on the computer that will use helpers: shows a code
    landschaft worker --join CODE     # on each helper, the first time
    landschaft worker                 # afterwards: wait for work until Ctrl+C
"""

from __future__ import annotations

import argparse
import sys

from ._dist import _group


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="landschaft", description="landschaft: lend this computer to the "
                                     "group of helpers, or manage the group (the analyses are run from Python).")
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
    parser.landschaft_commands = {"worker": wk, "helpers": hp}
    return parser


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
    else:
        parser.print_help()


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
