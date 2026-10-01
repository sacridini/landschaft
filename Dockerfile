# syntax=docker/dockerfile:1
#
# Landschaft in a container: the package built from this source tree, with
# every optional feature (dask, matplotlib, geopandas, pyarrow).
#
#   docker build -t landschaft .
#   docker run --rm -it -v "$PWD:/data" landschaft               # Python shell
#   docker run --rm -v "$PWD:/data" landschaft python script.py  # a script in the current folder
#
# Build arguments:
#   PYTHON_VERSION  Python of the image (default 3.12)
#   EXTRAS          optional dependencies to install (default "all"; "" for the core only)
#
# `docker build --target test -t landschaft:test .` builds an image that runs
# the test suite instead.

ARG PYTHON_VERSION=3.12

# ---- build: compile the C++ core into a wheel --------------------------------
FROM python:${PYTHON_VERSION}-slim AS build

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml CMakeLists.txt README.md LICENSE ./
COPY src ./src
# scikit-build-core fetches CMake, Ninja and pybind11; CMake fetches Eigen.
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---- base: the installed package -------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ARG EXTRAS=all

# libgomp1: the OpenMP runtime of the compiled core; libexpat1: needed by the
# GDAL bundled in the rasterio wheel.
RUN apt-get update  && apt-get install -y --no-install-recommends libgomp1 libexpat1  && rm -rf /var/lib/apt/lists/*

COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --find-links /wheels "landschaft${EXTRAS:+[$EXTRAS]}"  && rm -rf /wheels

ENV PYTHONUNBUFFERED=1     MPLBACKEND=Agg

RUN useradd --create-home --uid 1000 landschaft

# ---- test: the test suite against the installed package ----------------------
FROM base AS test

RUN pip install --no-cache-dir "pytest>=7" threadpoolctl
# The tests use the reference scripts in tools/ and the bundled maps by their
# path in the source tree.
COPY --chown=landschaft pyproject.toml /tests/pyproject.toml
COPY --chown=landschaft src/landschaft/data /tests/src/landschaft/data
COPY --chown=landschaft src/landschaft/bio/data /tests/src/landschaft/bio/data
COPY --chown=landschaft src/landschaft/cost/data /tests/src/landschaft/cost/data
COPY --chown=landschaft src/landschaft/surface/data /tests/src/landschaft/surface/data
COPY --chown=landschaft tools /tests/tools
COPY --chown=landschaft tests /tests/tests
USER landschaft
WORKDIR /tests
CMD ["pytest", "-q", "-p", "no:cacheprovider"]

# ---- runtime (default target) ------------------------------------------------
FROM base AS runtime

USER landschaft
WORKDIR /data
CMD ["python"]
