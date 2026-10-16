# SPDX-License-Identifier: GPL-3.0-or-later
"""Many independent tasks on this computer and on other computers of the local network.

A computer lends itself to the group with ``landschaft worker`` (the first time
``landschaft worker --join <code>``, with the code that ``landschaft helpers invite``
shows on the computer that will use it). Functions that run many independent tasks take
``on=ls.helpers()`` and run them here and on the helpers, with the same results as on
this computer alone::

    import landschaft as ls

    with ls.helpers() as pool:
        runs = ls.batch("fad", "tiles/", "fad_out/", on=pool)
        ens = ls.simulate.scenarios(model, start, drivers, replicates=200, seed=1, on=pool)

Helpers only run landschaft's own operations, never code sent to them: a task is the
name of an operation, its parameters and its input files. They connect out to the
computer that coordinates (only that one needs a firewall rule), authenticated by the
group's secret, over TLS.
"""

from __future__ import annotations

from ._dist import (
    Helpers,
    JobResult,
    helpers,
    invite_helpers,
    leave_group,
    list_helpers,
    open_firewall,
    run_worker,
)

__all__ = ["Helpers", "JobResult", "helpers", "invite_helpers", "leave_group", "list_helpers", "open_firewall",
           "run_worker"]
