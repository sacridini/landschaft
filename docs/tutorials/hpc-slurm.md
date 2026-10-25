# On a cluster (HPC, Slurm)

A cluster runs work as **batch jobs**: a script asks a scheduler (Slurm, PBS,
LSF) for processors, memory and time, and runs on a compute node when they are
free. Landschaft notices when it runs inside a job and adapts by itself, so the
script you run on your computer runs unchanged in a job: it uses the
processors the job was given, keeps its downloads where the nodes can reach
them, and does not try to open windows.

**You will learn to:**

- install Landschaft on a cluster, with pip or a container;
- run a script in a job, with every processor it asked for;
- split a folder of rasters among the tasks of a **job array**, follow it and
  submit again what failed;
- survive the time limit of a job;
- prepare the data for compute nodes without internet access;
- spread the tasks of one script over several nodes;
- find out what Landschaft sees on a node.

The commands are Slurm's, the most common scheduler; PBS and LSF work the same
way where the text says so.

## Install

On the cluster's **login node** (the computer you reach with `ssh`), in a
virtual environment kept in your home or project folder, which the compute
nodes see too:

<!-- no-run -->
```bash
module load python/3.12            # the name of the module differs between clusters
python -m venv ~/venvs/landschaft
source ~/venvs/landschaft/bin/activate
pip install landschaft
```

The wheels need glibc 2.28 or newer (Rocky/Alma Linux 8, Ubuntu 20.04, Debian
11 and later), on x86-64 or ARM64. On an older system, or to keep everything
in one file, use the container image with **Apptainer** (formerly
Singularity), which most clusters have:

<!-- no-run -->
```bash
apptainer pull landschaft.sif docker://ghcr.io/sacridini/landschaft:latest
apptainer exec landschaft.sif python analysis.py
apptainer exec landschaft.sif landschaft doctor
```

Apptainer runs as you and sees your home and the current folder; give it other
folders with `--bind /scratch`. Use a version instead of `latest`
(`landschaft:1.51.0`) to keep the results of a study reproducible.

## One job: a script on one node

A job script is a shell script with `#SBATCH` lines that say what the job
needs:

<!-- no-run -->
```bash
#!/bin/bash
#SBATCH --job-name=fragmentation
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/venvs/landschaft/bin/activate
python analysis.py
```

Submit it with `sbatch job.sh`; `squeue -u $USER` shows it waiting or running,
and its output goes to `slurm-<job>.out`. There is no `n_threads` to set:
inside a job every tool uses **all the processors of the job** (on a desktop it
keeps one free for you). Landschaft counts them from what the scheduler
confines the job to, or, on a cluster that does not, from
`SLURM_CPUS_PER_TASK` (`SLURM_CPUS_ON_NODE` without it; `NCPUS` on PBS,
`LSB_DJOB_NUMPROC` on LSF). A node of 128 cores running a job of
`--cpus-per-task=16` gives 16 threads, not 128.

What decides the number, from the first that applies:

| | |
|---|---|
| `n_threads=` | this call |
| `ls.set_default_threads(n)` | this process |
| `LANDSCHAFT_NUM_THREADS` | this process and the processes it starts |
| a dask.distributed worker | its threads |
| `OMP_NUM_THREADS` | as for any OpenMP program |
| the job | its processors |
| the computer | its processors, all but one |

Clusters often load modules that export `OMP_NUM_THREADS=1` (for MPI
programs), which would leave the job on one thread: Landschaft then warns once
and says how to change it (`export LANDSCHAFT_NUM_THREADS=$SLURM_CPUS_PER_TASK`).

## Many rasters: a job array

A folder of hundreds of tiles is better spread over many small jobs than run
in one large one: they start sooner, and a failure costs one part. A **job
array** is one script submitted as many tasks; each task gets its number in
`SLURM_ARRAY_TASK_ID`. `landschaft slurm` writes that script for a batch:

<!-- no-run -->
```bash
landschaft slurm fad tiles/ fad_out/ --cpus 16 --time 1:00:00
```

It counts the rasters, chooses the number of tasks (each task runs as many
rasters at once as it has processors), suggests the memory from the largest
raster and the tool, and prints the script it wrote to
`fad_out/.landschaft-batch/fad.sbatch`:

<!-- no-run -->
```bash
#SBATCH --job-name=landschaft-fad
#SBATCH --array=0-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=6G
#SBATCH --time=1:00:00
#SBATCH --output=.../fad_out/.landschaft-batch/logs/slurm-%A_%a.out
#SBATCH --signal=B:TERM@120
#SBATCH --requeue

exec python -m landschaft batch fad tiles fad_out --shard array --requeue --threads-per-worker 1 --max-memory 5120MiB
```

Submit it with `sbatch`, or add `--submit`. Parameters of the tool follow the
folders as `key=value` (`landschaft slurm mspa tiles/ out/ edge_width=2`);
`--partition`, `--account`, `--tasks`, `--max-running`, `--sbatch` (any other
`#SBATCH` option) and `--setup` (a line run before Landschaft, such as
`module load ...`) shape the script.

Each task runs the same command, `landschaft batch ... --shard array`, and
processes its part of the rasters: in the order of the folder, raster `i` goes
to part `i mod n`. The split is kept in the output folder, so the tasks you
submit again later take the same rasters. From Python, the same is
`ls.batch(tool, inputs, output, shard="array")`, and outside an array that
processes every raster, so the script also runs on your computer:

```python
from pathlib import Path
import landschaft as ls

forest = ls.load_raster(ls.example_data("input"))
Path("tiles").mkdir(exist_ok=True)
for row in range(0, 1000, 250):
    ls.write_raster(forest[row:row + 250, :500], f"tiles/forest_{row}.tif")

runs = ls.batch("accounting", "tiles", "accounting_out", shard="array")   # every raster here
print(len(runs))
```

`shard="i/n"` gives a part by hand (part `i`, from 0, of `n`). PBS and LSF
arrays do not tell their tasks how many there are, so there it is
`--shard $PBS_ARRAY_INDEX/50` (or `$LSB_JOBINDEX`, counting from 0):

```python
first = ls.batch("accounting", "tiles", "parts_out", shard="0/2")   # rasters 0 and 2
print([run.input.name for run in first])
```

!!! note "Why not automatic"
    `ls.batch` takes its part only when asked (`shard="array"`): a script that
    runs a batch in every task of an array for another reason, a scenario per
    task for example, would otherwise process a fraction of its rasters
    without a word.

## Follow it and submit again what is left

Rasters already done are never computed again (each one's `.json`, or MSPA's
`.tif`, marks it done, written only once it is whole). `landschaft status`
reads the record the tasks keep in the output folder:

<!-- no-run -->
```bash
$ landschaft status fad_out/
fad in fad_out: 400 inputs, 391 done, 2 failed, 7 not done
  failed: tile_0137.tif (run 4567_12): MemoryError: ...
  not started: tile_0390.tif, tile_0391.tif, ...
To process the 9 left: sbatch --array=12,15-17 fad_out/.landschaft-batch/fad.sbatch
```

The last line submits only the tasks that have rasters left. The exit code is
0 when every raster is done and 3 otherwise, for scripts that chain jobs. From
Python:

```python
status = ls.batch_status("parts_out")
print(status.done, status.left)
```

## The time limit

Slurm stops a job at its time limit. With `--signal=B:TERM@120` (in the scripts
`landschaft slurm` writes) it warns two minutes before with SIGTERM: the
batch then starts no new raster, lets the ones at work end, and with
`--requeue` puts the task back in the queue to continue (`scontrol requeue`),
if it finished at least one raster in this run (a raster longer than the time
limit would otherwise come back forever). A job you cancel with `scancel` is
not requeued. In Python, the stop is `ls.batch.Stopped`, which holds the
results of the rasters that ended.

## Nodes without internet access

Compute nodes often cannot reach the internet. Landschaft downloads a few
things on first use: the reference data of species occurrence cleaning
(`ls.sdm.reference_data`) and some example data. Download them on the login
node, into the cache that the compute nodes see too:

<!-- no-run -->
```bash
landschaft fetch          # all; or: landschaft fetch sdm, landschaft fetch examples
```

On a node without access, a download says so and names the `landschaft fetch`
that gets it, instead of a network error. The cache is `~/.cache/landschaft`;
when the home quota is small, point it at the scratch space with
`export LANDSCHAFT_CACHE=/scratch/$USER/landschaft-cache` (in the job script
too). Occurrence records from GBIF (`ls.sdm.download_gbif`) also need the
internet: download them on the login node and save them as a table.

Plots cannot open a window in a job: `ls.plot(...)` says so once, and
`.save("map.html")` or `.save("map.png")` writes them.

## Several nodes for one script

The functions that run many independent tasks (`batch`, `simulate.scenarios`,
`simulate.calibrate`, `sdm.fit`, `monte_carlo`, `parallel.map`...) take
`on=ls.helpers()` and share their tasks with
[helper computers](many-computers.md). In an allocation of several nodes, the
other nodes become helpers of the first:

<!-- no-run -->
```bash
#!/bin/bash
#SBATCH --nodes=4 --ntasks-per-node=1 --cpus-per-task=32 --time=04:00:00

source ~/venvs/landschaft/bin/activate
srun --nodes=3 --ntasks=3 --cpus-per-task=32 --exclude="$SLURMD_NODENAME" landschaft worker --connect "$SLURMD_NODENAME" &
python scenarios.py        # with ls.helpers() as pool: ... on=pool
```

Inside a job the coordinator does not look for helpers by broadcast (it does
not pass between nodes), and `landschaft worker` needs `--connect`. The
helpers wait for the group that `ls.helpers()` creates in the configuration
folder of your home, which every node sees. The results are the same as on one
computer. A job array is simpler when the tasks are rasters of a folder; the
helpers are for the other many-task functions.

## What Landschaft sees

`landschaft doctor` prints what Landschaft finds on the computer it runs on:
run it in a job to check the threads, and paste it in a question or an issue:

<!-- no-run -->
```bash
$ srun --cpus-per-task=8 landschaft doctor
landschaft 1.51.0 (Python 3.12.4, Linux x86_64, glibc 2.34)
  compiled core: _core.cpython-312-x86_64-linux-gnu.so, OpenMP yes, AVX2+FMA yes
Threads (the default of n_threads): 8 threads (8 of 128 processors available to this process; Slurm job 4567)
Job: Slurm job 4567 on node042
Cache: /home/me/.cache/landschaft (default), writable
Network:
  GBIF (sdm.download_gbif)               NO: the name does not resolve (no DNS)  (api.gbif.org)
  ...
  ! This computer seems to have no internet access ...: `landschaft fetch` downloads the data ...
Display: none, inside a Slurm job: plots are not shown, .save() writes them
```

From Python, `ls.doctor()`:

```python
ls.doctor(network=False)
```

## Troubleshooting

| What you see | Why, and what to do |
|---|---|
| a warning about `OMP_NUM_THREADS=1` | a module exported it: `export LANDSCHAFT_NUM_THREADS=$SLURM_CPUS_PER_TASK` |
| the job is slower than the desktop | it got fewer processors: ask with `--cpus-per-task`, check with `landschaft doctor` |
| `Disk quota exceeded` in the cache | `export LANDSCHAFT_CACHE=/scratch/$USER/landschaft-cache` |
| `could not reach ... no internet access` | `landschaft fetch` on the login node |
| `GLIBC_2.28 not found` when importing | the system is older than the wheels: use the Apptainer image |
| a task killed with `OUT_OF_MEMORY` | more `--mem`, fewer `--cpus` per task, or `--max-memory` lower |
| tasks of a resubmitted array take other rasters | the split changed: `landschaft slurm` warns; delete `.landschaft-batch/split.json` only to start over |
