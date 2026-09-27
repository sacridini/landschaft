# Installation

Landschaft is a Python package with a compiled C++ core. Until binary wheels
are published it is built from source, which takes two to five minutes the
first time.

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.10 | 3.12 recommended |
| C++ compiler | C++17 | MSVC 2019+ (Windows), GCC 9+ (Linux), Clang 10+ (macOS) |
| CMake | 3.20 | installed automatically by the build if missing |
| OpenMP | optional | without it everything runs on one thread |
| Memory | 4 GB | larger maps need more, or the out-of-core options |

The Python dependencies (numpy, pandas, scipy, xarray, rasterio, rioxarray,
pybind11, scikit-build-core) are installed by the build.

## Install with conda (recommended)

[conda](https://docs.conda.io) (or its faster drop-in, mamba) provides Python
and the geospatial libraries (GDAL behind rasterio) as tested binaries on every
platform, which avoids most installation problems.

=== "Windows"

    1. Install **Miniforge** from <https://conda-forge.org/download/>.
    2. Install the **Build Tools for Visual Studio** from
       <https://visualstudio.microsoft.com/visual-cpp-build-tools/> and tick
       *Desktop development with C++*.
    3. Open the *Miniforge Prompt* and run:

    ```bat
    conda create -n landschaft -c conda-forge python=3.12 numpy pandas scipy rasterio rioxarray ^
          xarray dask matplotlib geopandas scikit-build-core pybind11 cmake ninja
    conda activate landschaft
    git clone https://github.com/sacridini/landschaft.git
    cd landschaft
    pip install --no-build-isolation -e .
    ```

    !!! tip "Only the Build Tools installed?"
        If CMake does not find the compiler, name the Visual Studio generator
        before `pip install`: `set CMAKE_GENERATOR=Visual Studio 17 2022`.

=== "macOS"

    1. Install the command-line tools: `xcode-select --install`.
    2. Install **Miniforge** from <https://conda-forge.org/download/>.
    3. In a terminal:

    ```bash
    conda create -n landschaft -c conda-forge python=3.12 numpy pandas scipy rasterio rioxarray \
          xarray dask matplotlib geopandas scikit-build-core pybind11 cmake ninja llvm-openmp
    conda activate landschaft
    git clone https://github.com/sacridini/landschaft.git
    cd landschaft
    pip install --no-build-isolation -e .
    ```

    `llvm-openmp` gives Apple's compiler the OpenMP runtime it lacks; without
    it Landschaft builds and runs, single-threaded.

=== "Linux"

    1. Install a compiler: `sudo apt install build-essential` (Debian,
       Ubuntu) or `sudo dnf install gcc-c++` (Fedora).
    2. Install **Miniforge** from <https://conda-forge.org/download/>.
    3. In a terminal:

    ```bash
    conda create -n landschaft -c conda-forge python=3.12 numpy pandas scipy rasterio rioxarray \
          xarray dask matplotlib geopandas scikit-build-core pybind11 cmake ninja
    conda activate landschaft
    git clone https://github.com/sacridini/landschaft.git
    cd landschaft
    pip install --no-build-isolation -e .
    ```

The `-e` (editable) flag makes `git pull` followed by the same `pip install`
line enough to update; the C++ core is rebuilt only where it changed.

## Install with pip only

With a system Python and a compiler already installed, a virtual environment
is enough (rasterio then comes as a binary wheel that bundles GDAL):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "landschaft[all] @ git+https://github.com/sacridini/landschaft.git"
```

## Run with Docker

The repository has a `Dockerfile` that builds Landschaft with every optional
feature in a Linux image, so the host needs neither a compiler nor Python, only
[Docker](https://docs.docker.com/get-docker/):

```bash
git clone https://github.com/sacridini/landschaft.git
cd landschaft
docker build -t landschaft .
```

The container works in `/data`; mount the folder with your maps and scripts
there:

```bash
docker run --rm -it -v "$PWD:/data" landschaft                 # Python shell
docker run --rm -v "$PWD:/data" landschaft python analysis.py  # run a script
```

In PowerShell write `${PWD}` instead of `$PWD`. Outputs written with relative
paths land in the mounted folder. Other options:

| Command | Effect |
|---|---|
| `docker build --build-arg PYTHON_VERSION=3.11 -t landschaft .` | another Python version |
| `docker build --build-arg EXTRAS= -t landschaft .` | the core only, without dask, matplotlib and geopandas (smaller image) |
| `docker build --target test -t landschaft:test .` then `docker run --rm landschaft:test` | run the test suite in the container |
| `docker run --cpuset-cpus 0-7 ...` | limit the cores; Landschaft then uses all of them but one |

## Optional features

Some features need extra packages. Install them with conda, or as *extras*
with pip (`pip install -e ".[plot,vector]"`):

| Extra | Packages | Needed for |
|---|---|---|
| `plot` | matplotlib | the `show_*` plotting functions and the figures in the tutorials |
| `vector` | geopandas, pyarrow | vector inputs and outputs (sample plots, zones, habitat networks from polygons), Parquet tables |
| `dask` | dask | lazy and parallel processing of stacks and very large arrays |
| `all` | all of the above | |
| `docs` | mkdocs-material, mkdocstrings | building this website |
| `test` | pytest, scipy, dask | running the test suite |

## Check the installation

```python
import landschaft as ls

print(ls.__version__)        # the installed version
print(ls.max_threads())      # processors the C++ core can use
print(ls.default_threads())  # threads used by default: all of them but one

forest = ls.load_raster(ls.example_data("input"))
print(ls.mspa_stats(ls.mspa(forest)))
```

If the last line prints a table of MSPA classes, everything works. To run the
whole test suite (about four minutes):

```bash
pip install -e ".[test]"
pytest
```

## Troubleshooting

??? question "`pip install` fails with *no C++ compiler found* / *CMAKE_CXX_COMPILER not set*"
    The build needs a C++17 compiler: the *Desktop development with C++*
    workload of the Visual Studio Build Tools on Windows, `xcode-select
    --install` on macOS, `build-essential` or `gcc-c++` on Linux. On Windows,
    also try `set CMAKE_GENERATOR=Visual Studio 17 2022`.

??? question "`ls.max_threads()` returns 1"
    The core was built without OpenMP. On macOS install `llvm-openmp` (conda)
    or `libomp` (Homebrew) and reinstall; with GCC and MSVC OpenMP is always
    available.

??? question "`ImportError: DLL load failed` / `rasterio` cannot find GDAL"
    Mixing conda packages and pip wheels of GDAL-based libraries breaks them.
    Create a fresh conda environment with the command above and install
    rasterio, rioxarray and geopandas from conda-forge only.

??? question "The build uses too much memory or is slow"
    Pass `-Cbuild-dir=build` to `pip install` to keep the build directory:
    later installs only recompile what changed.
