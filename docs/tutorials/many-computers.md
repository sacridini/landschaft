# Several computers

A batch of a thousand tiles, two hundred replicates of a land change model, a
calibration grid of fifty points: each task is independent of the others, so
the computers of a lab can share them. Landschaft does this without a cluster:
each computer that lends itself runs one command, and the functions that run
many tasks take `on=` and give the same results as on one computer.

**You will learn to:**

- bring computers into a group, once, with a short code;
- lend a computer to the group, with all its cores or a part;
- send the tasks of `batch`, `mspa_batch`, `simulate.scenarios` and
  `simulate.calibrate` to the group;
- know what travels between computers, and what does not;
- find out why a helper does not show up.

This helps when there are **many independent tasks**. One large map is better
served by the threads of one computer (every tool already uses them) or by
[tiles](large-data.md#mspa-of-maps-larger-than-memory).

## Bring computers into a group

On the computer that will hand out the work, open an invitation:

<!-- no-run -->
```bash
landschaft helpers invite
```

It prints a code such as `K7QM-2XDP`, valid for ten minutes. On each computer
that will help, type it once:

<!-- no-run -->
```bash
landschaft worker --join K7QM-2XDP
```

The code only authenticates this first meeting: in it the helper receives the
group's secret (random, 256 bits), kept in the user's configuration folder on
both computers. From then on there is nothing to type but `landschaft worker`.
Every computer with Landschaft installed has the command (`python -m landschaft`
does the same).

## Lend a computer

<!-- no-run -->
```bash
landschaft worker              # all cores but one, until Ctrl+C
landschaft worker --cores 6    # lend only part of the machine
```

The helper waits in the background. When a computer of its group has work,
the helper finds it on the local network, takes tasks, runs them and sends the
outputs back; when the work ends it goes back to waiting. The computer does not
go to sleep while a task runs.

## Send the tasks to the group

`ls.helpers()` is this computer and the helpers of its group. Give it to `on=`:

<!-- no-run -->
```python
import landschaft as ls

with ls.helpers() as pool:
    runs = ls.batch("fad", "tiles/", "fad_out/", window_size=27, on=pool)
    ens = ls.simulate.scenarios(model, start, drivers, replicates=200, seed=1, on=pool)
    cal = ls.simulate.calibrate(build, start, observed, params={"w": (0.0, 1.0)},
                                method="random", max_evals=60, seed=1, on=pool)
```

Inside the `with` block the helpers stay connected from one call to the next;
`on=ls.helpers()` in a single call works too. This computer also runs tasks,
with all its cores but one (`local_threads=` changes that); with
`ls.helpers(local=False)` it only hands them out and gathers the outputs.

Helpers pull the work: each asks for a task when it has room, so a fast and a
slow computer balance by themselves. The heaviest tasks go first, a task that
fails is tried again on another computer, a helper that disappears gives its
tasks back, and near the end an idle computer also runs a task that a slow one
is still on: the first result counts. A line shows the progress:

<!-- no-run -->
```text
+ lab-pc-3 (192.168.1.23, 15 cores) joined
scenarios: 142/200 | 4 computers | ~1 min 10 s left
scenarios: 200 task(s) in 3 min 02 s
  lab-main (this computer): 61, lab-pc-3: 70, lab-pc-5: 38, lab-pc-7: 31
```

| Function | One task | Notes |
|---|---|---|
| `ls.batch` | one raster through one tool | `tool` by name (one of `ls.BATCH_TOOLS`), parameters as numbers, strings, lists or dicts |
| `ls.mspa_batch` | MSPA of one raster | |
| `ls.simulate.scenarios` | one run of one scenario | each scenario's model travels as its graph (`Model.save`) |
| `ls.simulate.calibrate` | one point of a `"grid"` or `"random"` search, with its replicates | built-in targets; the best point's simulation is run again here |

## The same results

A run on the group gives the same outputs as on one computer: the seeds of the
replicates are set here (`seed + r`), the outputs are put together in the order
of the tasks, not of their arrival, and Landschaft's results never depend on the
number of threads. A helper with another version of Landschaft, or the same
version with other code (a development install), gets no tasks until both are
the same; the coordinator says so.

## What travels

A task is the **name of an operation** of Landschaft, its parameters (JSON) and
its input files. A helper runs only Landschaft's own operations, from a fixed
list, never code sent to it: lending a computer does not give anyone a way to
run programs on it. Parameters that name files must stay inside the task's own
folder. Helpers only connect out, to the computer that hands out the work;
every connection is encrypted (TLS) and both sides prove they know the group's
secret.

Input files reach a helper in the cheapest way it has:

1. **its cache**: each file is kept by the hash of its bytes (20 GB by
   default, `--cache-gb`), so the same data is never sent twice;
2. **a shared folder** (a NAS) that both computers see: the helper reads it
   directly. A mapped Windows drive is recognised by its network path; on Linux
   or macOS, say where the share is mounted: `landschaft worker --map
   Z:/=/mnt/nas`;
3. **the network**, from the computer that hands out the work.

The outputs of `batch` and `mspa_batch` go straight into the output folder when
the helper sees it too (on the NAS), otherwise back over the network. An
interrupted `batch` continues where it stopped: the files already done are
skipped.

## When a helper does not show up

- `landschaft helpers list` (on any computer of the group) shows the helpers
  waiting for work.
- Only the computer that hands out the work needs to accept connections. On
  Windows, `landschaft helpers firewall` creates the rules (TCP 47811, UDP
  47810, local network only); on Linux allow those ports in the firewall.
- The helpers find the coordinator by a broadcast on the local network. Over a
  VPN, another subnet or a Wi-Fi that drops broadcasts, point them at it:
  `landschaft worker --connect 192.168.1.10`.
- Wi-Fi is slow for large inputs: the coordinator warns when the helpers spend
  more time waiting for their files than working. A NAS or a cable helps more
  than more helpers.
- `landschaft helpers leave` takes a computer out of its group.

## Next steps

- The [API reference](../api/batch.md#several-computers) lists every option.
- [Large data & batch](large-data.md) covers the work of one computer.
