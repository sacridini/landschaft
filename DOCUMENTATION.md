# landschaft: Complete Documentation & User Guide

**landschaft** is a high-performance Python package reimplementing the complete scientific raster analysis suite of [GuidosToolbox (GTB)](https://forest.jrc.ec.europa.eu/en/activities/lpa/gtb/) developed by the European Commission Joint Research Centre (JRC).

It provides a native **C++17 / OpenMP** core exposed through zero-copy **pybind11** wrappers, offering high-level Python APIs fully integrated with **numpy**, **xarray**, **rioxarray**, **pandas**, and **dask**.

---

## Table of Contents

1. [Architecture & Design Principles](#1-architecture--design-principles)
2. [Installation & Requirements](#2-installation--requirements)
3. [Quickstart & Input Conventions](#3-quickstart--input-conventions)
4. [Raster I/O, Reclassification & Palettes](#4-raster-io-reclassification--palettes)
5. [Morphological Spatial Pattern Analysis (MSPA)](#5-morphological-spatial-pattern-analysis-mspa)
6. [Topological Network Analysis (GTN & Conefor)](#6-topological-network-analysis-gtn--conefor)
7. [Fragmentation & Distance Transforms (FAD, Distance, Entropy)](#7-fragmentation--distance-transforms-fad-distance-entropy)
8. [Landscape Mosaic & Contagion/Clustering (LM, Pff, FAC)](#8-landscape-mosaic--contagionclustering-lm-pff-fac)
9. [Restoration & Morphological Shape (RSS, Cost Distance, Corridors, Parcellation, Contortion)](#9-restoration--morphological-shape-rss-cost-distance-corridors-parcellation-contortion)
10. [Change Detection & Spatial Convolution (Simple Change, MCD, FOS Change, GraySpatCon)](#10-change-detection--spatial-convolution-simple-change-mcd-fos-change-grayspatcon)
11. [Batch Processing & Raster Collections](#11-batch-processing--raster-collections)
12. [Accuracy & Bit-for-Bit Verification](#12-accuracy--bit-for-bit-verification)
13. [Landscape Metrics (FRAGSTATS / landscapemetrics)](#13-landscape-metrics-fragstats--landscapemetrics)

---

## 1. Architecture & Design Principles

* **Line-by-line Mathematical & Logical Parity**: Reimplemented directly from GuidosToolbox C, IDL, and miallib sources (`fsp.c`, `spatcon.c`, `GraySpatCon.c`, `recode.c`, `guidostoolbox.pro`).
* **Maximum Performance via C++ and OpenMP**: Heavy raster operations (sliding windows, geodesic reconstructions, topological graph searches, Euclidean distance transforms) run in C++17 with multi-threaded OpenMP execution and GIL release.
* **$O(1)$ Sliding Window Algorithms**:
  * Forest Area Density (FAD) uses **2D Integral Images** (Summed-Area Tables) for $O(1)$ computation per pixel regardless of window size ($7 \times 7$ up to $243 \times 243$).
  * GraySpatCon uses **101-bin sliding histograms** that update columns in $O(1)$ rather than re-scanning the window.
* **Full Geospatial Metadata Preservation**: All high-level functions accept and return `xarray.DataArray` objects, seamlessly preserving CRS, affine geotransforms, spatial coordinates, and nodata masks.

---

## 2. Installation & Requirements

### System Requirements
* Python 3.10, 3.11, or 3.12+
* C++17 compliant compiler (GCC 9+, Clang 10+, or MSVC 2019+)
* CMake >= 3.20
* OpenMP (optional, automatically detected for parallel multithreading)

### Installing from Source

```bash
# Clone the repository
git clone https://github.com/sacridini/landschaft.git
cd landschaft

# Install dependencies and package in editable mode
pip install .
```

### With Conda / Mamba

```bash
conda create -n landschaft -c conda-forge python=3.12 numpy pandas rasterio rioxarray xarray dask \
      scikit-build-core pybind11 cmake ninja pytest scipy
conda activate landschaft
pip install --no-build-isolation -e .
```

---

## 3. Quickstart & Input Conventions

GuidosToolbox standard binary format uses single-band `uint8` rasters:
* `0`: Missing / Nodata (ignored in analysis)
* `1`: Background (matrix / non-habitat)
* `2`: Foreground (focal habitat / forest)

```python
import landschaft as ls

# 1. Load sample dataset or GeoTIFF
data = ls.load_raster(ls.example_data("input"))  # 1000x1000 binary map

# 2. Run MSPA
mspa_map = ls.mspa(data, connectivity=8, edge_width=1)

# 3. Calculate Fragmentation (FAD)
fad_map = ls.fad(data, window_size=27)

# 4. Save to GeoTIFF with original GTB colormap
ls.write_raster(mspa_map, "output_mspa.tif")
ls.write_raster(fad_map.fad_map, "output_fad.tif")
```

---

## 4. Raster I/O, Reclassification & Palettes

### `load_raster`
Loads a GeoTIFF or raster file as a georeferenced `xarray.DataArray`. If passed a folder or glob pattern, returns a `RasterCollection`.
```python
# Single file
da = ls.load_raster("landcover.tif", validate="mspa")  # checks 0/1/2 coding
raw = ls.load_raster("dem.tif", validate=None)         # loads raw continuous data

# Multiple files / folder
collection = ls.load_raster("tiles/*.tif")
```

### `reclassify`
Conveniently reclassifies any landcover classification map into GTB binary coding (`0`, `1`, `2`):
```python
clc = ls.load_raster("corine.tif", validate=None)
# Classes 23, 24, 25 are forest (foreground=2), nodata 48 is missing=0, others background=1
forest = ls.reclassify(clc, foreground=[23, 24, 25], missing=[48])

# Using lambda condition:
dense = ls.reclassify(tree_cover, foreground=lambda arr: arr >= 40)
```

### `write_raster`
Writes a 2D array or `xarray.DataArray` to a GeoTIFF with optional LZW/Deflate compression and the official GuidosToolbox colormap:
```python
ls.write_raster(result_da, "output.tif", colormap="auto")
```

### Available Colormaps (`landschaft.io`)
Extracted bit-for-bit from original GTB `.sav` colortables:
* `mspa_colormap(transition=True)`
* `gtn_components_colormap()`
* `gtn_importance_colormap()`
* `fad_colormap()`
* `fad5_colormap()` (5-class FAD/FOS)
* `dist_colormap()`
* `entropy_colormap()`
* `lm_colormap()`
* `contortion_colormap()`
* `resistance_colormap()`
* `change_colormap()`
* `foschange_colormap()`

---

## 5. Morphological Spatial Pattern Analysis (MSPA)

MSPA segments foreground habitat into 7 mutually exclusive morphological classes and openings based on geometric connectivity and distance to boundaries.

```python
result = ls.mspa(
    data,
    connectivity=8,   # Foreground connectivity: 8 (default) or 4
    edge_width=1,     # Effective edge width in pixels (1 to 255)
    transition=True,  # Transition parameter for color styling
    internal=True,    # Identifies internal perforations and core openings
    n_threads=0,      # Number of OpenMP threads (0 = all available cores)
)
```

### MSPA Tiling (maps larger than memory)
GTB's *MSPA Tiling*: overlapping tiles processed one by one (or `n_workers` at a time) and put back together.
A GeoTIFF path is processed out of core, reading and writing one tile window at a time:
```python
out = ls.mspa_tiled("europe_forest.tif")        # -> europe_forest_8_1_1_1.tif (GTB naming)
out = ls.mspa_tiled("europe_forest.tif", "mspa.tif", connectivity=4, n_workers=4, overwrite=True)
m = ls.mspa_tiled(data, tile_size=6500, overlap=2000)  # arrays / DataArrays in memory
ls.tile_layout(20000)  # [(read_start, read_stop, keep_start, keep_stop), ...] per tile along an axis
```
Tiles are `tile_size` pixels including the `overlap` (GTB: 6500 and 2000 on MS-Windows, 9500 on Linux,
8500 on macOS; GTB fixes the edge width to 1); each tile keeps its half of every overlap. Tiles with only
missing data, only foreground, or lacking background or foreground are classified directly, as in GTB.
The result is identical to GTB's MSPA Tiling. It equals MSPA of the whole map when the objects are small
compared with the overlap; otherwise classes that depend on whole objects (loops, bridges, islets, openings)
can differ near the seams — use `ls.mspa()` on the whole map whenever it fits in memory (about 32 bytes per pixel).

### Simplified Pattern Analysis (SPA2/3/5/6)
GTB's simplified alternatives to MSPA (8-connected foreground, 1-pixel edge width):
```python
r = ls.spa(data, variant=5)   # 2, 3, 5 or 6
r.map       # 17 core (SPA2: contiguous), 1 margin (SPA2: small linear features), 3 edge,
            # 5 perforation, 9 islet (SPA6), 100 core-opening, 0 background, 129 missing
r.stats     # area, % of data, % of foreground per class
r.counts, r.porosity
print(r.summary())
```
Identical to GTB (maps and statistics).

### MSPA Classes Table (`ls.MSPA_CLASSES`)

| Pixel Value | Class Name | Description |
| :---: | :--- | :--- |
| **17** | **Core** | Interior habitat area beyond edge width distance from perimeter. |
| **9** | **Islet** | Disconnected habitat patch too small to contain core. |
| **5** | **Perforation** | Internal boundary pixels surrounding interior openings. |
| **3** | **Edge** | External perimeter boundary pixels of habitat patches. |
| **33 / 35 / 37** | **Bridge** | Structural corridor connecting two or more distinct Core patches. |
| **65 / 67 / 69** | **Loop** | Linear corridor connecting back to the same Core patch. |
| **1** | **Branch** | Linear corridor connected to Core at one end and terminating in matrix. |
| **100** | **Core-opening** | Background perforation completely enclosed within Core. |
| **220** | **Border-opening** | Background perforation touching the external boundary. |
| **+100** | **Internal feature** | Any feature situated inside an interior perforation (`internal=True`). |

### Class Statistics
```python
df_stats = ls.mspa_stats(result)
print(df_stats)
```

---

## 6. Topological Network Analysis (GTN & Conefor)

Identifies functional components and ranks the structural importance of individual habitat nodes and corridors.

```python
from landschaft import gtn_components, gtn_importance, conefor_inputs

# Step 1: Run MSPA to get segmented classes
mspa_map = ls.mspa(data, edge_width=2)

# Step 2: Extract connected network components
net = gtn_components(mspa_map, min_patch_size=1)
print(net.summary())
# Output includes:
# - Equivalent Connected Area (ECA)
# - Degree of Coherence (DOC [%])
# - Number of network components and isolated cores

# Step 3: Compute node and link importance (Probability of Connectivity - dPC)
importance = gtn_importance(mspa_map)
print(importance.node_table.head())
print(importance.link_table.head())
# Metrics include: dPC, dPCintra, dPCflux, dPCconnector

# Step 4: Export to Conefor Sensinode
conefor_inputs(mspa_map, output_dir="./conefor_files")
# Generates nodes.txt and distances.txt
```

---

## 7. Fragmentation & Distance Transforms

### Forest Area Density (FAD & FAD Multiscale)
Calculates foreground pixel proportion within square moving windows.
```python
# Single window size
fad_res = ls.fad(data, window_size=27)
print(f"Average FAD: {fad_res.average_fad:.2f}%")

# Multiscale FAD across 5 default GTB window scales: 7, 13, 27, 81, 243
multi = ls.fad_multiscale(data, window_sizes=[7, 13, 27, 81, 243])
# multi.single_scales maps window_size -> DataArray; multi.multiscale_map is their rounded mean
```
Inputs may also contain GTB's special backgrounds: `3` (special background, coded 105 in the
maps) and `4` (non-fragmenting special background, treated as missing when measuring density,
coded 106). `multi.stats` lists the class shares, `average_fad` and GTB's `avcon`
(average FAD × foreground / non-missing area) per scale. The maps are identical to GTB's FAD-MS
outputs (`*_fad_<window>.tif`, `*_fad_mscale.tif`).
**FAD Classes (`ls.FAD_CLASSES`)**:
1. Rare ($0..9\%$)
2. Patchy ($10..39\%$)
3. Transitional ($40..59\%$)
4. Dominant ($60..89\%$)
5. Interior ($90..100\%$)

### Fragmentation at a Fixed Observation Scale (FOS)
GTB's *Fragmentation → FOS*, with all its methods and inputs:
```python
r = ls.fos(data, window_size=27, method="FAD", classes=6)   # "FAD", "FED" or "FAC"; 5 or 6 classes
r.map          # 0-100 connectivity of each foreground pixel; 101 BG, 102 missing, 105/106 special BG
r.stats        # share [%] of the foreground per class (Rare .. Interior/Intact)
r.histogram    # share [%] of the foreground pixels with each value 0..100
r.average, r.avcon, r.n_patches, r.average_patch_size, r.eca, r.coherence, r.coherence_ru
print(r.summary())  # GTB's fos.txt report

gray = ls.load_raster("cover.tif", validate=None)  # 0-100 % cover, 255 missing, 103/104 special BG
r = ls.fos(gray, 13, "FED", 5, gray_threshold=30)   # foreground = cover >= 30
```
* **FAD** (area density): share of foreground in the window (spatcon, or GraySpatCon metric 1 — the mean cover — for grayscale maps).
* **FED** (edge density): GraySpatCon metric 52 ("clustering") of the window's pixel pairs.
* **FAC** (area clustering): spatcon's foreground-foreground adjacency share; binary maps only, as in GTB.

Six classes separate *Intact* (100) from *Interior* (90-99); five merge them (colour table `fadcolors5`).
A map is taken as grayscale when its maximum exceeds 4 (as in GTB) unless `grayscale=` is given. Patch statistics
use `connectivity` (8 or 4). Maps and statistics are identical to GTB's batch FOS.

### Euclidean Distance Transform
Computes exact Signed Euclidean Distance transform using Meijster's $O(N)$ algorithm:
```python
dist_res = ls.euclidean_distance(data)
# dist_res.distance_float: continuous distance (positive inside FG, negative inside BG)
# dist_res.distance_byte : 256-level byte-scaled distance map for GIS visualization
# dist_res.mean_fg_dist  : average distance from perimeter to interior
# dist_res.max_fg_dist   : maximum distance to edge
```

### Influence Zones and Proximity (experimental)
GTB's *Distance → Influence Zones / Proximity*. The objects (8-connected foreground
areas of at least `min_area` pixels, with their holes filled) divide the map into the
zones of the watershed of the distance to the objects:
```python
iz = ls.influence_zones(data, min_area=5000, core_buffer=0, buffer=10)
iz.classes, iz.ids, iz.zones          # GTB viewport codes, object ids, zone labels
px = ls.proximity(data, min_area=5000)
px.distance, px.cag, px.lines         # proximity map, connected area gain, watershed pixels
px.to_csv("input_proximity_5000_0_0.csv")   # GTB's proximity CSV
```
These tools are **not yet identical to GTB**: IDL's `MORPH_DISTANCE`, which GTB uses, does
not always pick the nearest object next to the boundaries between zones, and its exact
rule is still unknown. The objects, ids, holes, buffers and CAG follow GTB's code
line by line.

### Spatial Joint Directional Entropy
Measures textural heterogeneity and spatial order across 4 directions ($0^\circ, 45^\circ, 90^\circ, 135^\circ$):
```python
# Global entropy indices
ent_res = ls.spatial_entropy(data)
print(f"Spatial Entropy H: {ent_res.entropy:.4f}")
print(f"Relative Spatial Entropy H_rel: {ent_res.relative_entropy:.4f}")

# Moving-window local entropy map
ent_map = ls.spatial_entropy_map(data, window_size=50)
```

---

## 8. Landscape Mosaic & Contagion/Clustering

Analyzes spatial mixing between three landcover components:
1. **Natural (N)**
2. **Agriculture (A)**
3. **Developed / Antropic (D)**

### Landscape Mosaic (LM)
```python
# 19-class or 103-class Landscape Mosaic
lm19 = ls.landscape_mosaic(data, window_size=27, output_classes=19)
lm103 = ls.landscape_mosaic(data, window_size=27, output_classes=103)

# 103-class barycentric statistics breakdown
stats = ls.landscape_mosaic_stats(lm103)
print(stats)

# GTB heatmap: share [%] of each of the 103 classes, as GTB stores it (single precision)
heat = ls.landscape_mosaic_heatmap(data, window_size=27)
heat.table          # share per class code, with its 19-class recode
heat.triangle       # 12 x 21 grid in GTB's triangle layout (NaN outside)
heat.to_csv("forest_lm_27_heatmap.csv")   # identical to GTB's heatmap.csv
```

### Contagion ($P_{ff}$) & Area Clustering (FAC)
```python
# Pff: pixel co-occurrence probability of foreground
pff = ls.contagion_pff(data, window_size=7)

# FAC: Foreground Area Clustering normalized by density (Pff / Pf)
fac = ls.clustering_fac(data, window_size=7)
```

---

## 9. Restoration & Morphological Shape

### Restoration Status (RSS)
Evaluates landscape connectivity before and after hypothetical restoration of all non-core pixels:
```python
rss = ls.restoration_status(mspa_map)
print(rss.summary())
# Attributes:
# - current_eca, potential_eca
# - doc_current, doc_potential
# - cnoa (Critical Nodes of Attachment)
# - rac (Restorable Area of Connectivity)
```

### Cost Distance & Least-Cost Corridors
Anisotropic 8-neighbor Dijkstra least-cost routing across resistance surfaces:
```python
# 1. Cost Distance Map from Source Mask
cost_map = ls.cost_distance(resistance_surface, source_mask)

# 2. Two-ended Least Cost Path
lcp = ls.least_cost_path(resistance_surface, marker_a, marker_b)
print(f"Total Path Cost: {lcp.total_cost:.2f}, Length: {lcp.path_length} pixels")

# 3. Restoration Corridor (Dijkstra forward-backward intersection)
corridor = ls.restoration_corridor(resistance_surface, marker_a, marker_b, corridor_width=3)
```

### Accounting (Object Size Classes)
GTB's *Objects → Accounting*: labels every foreground object and groups it into
up to six size classes. The maps are identical to GTB's `_acc.tif`,
`_acc_ids.tif` and `_acc_pixels.tif`, and the numbers match its `acc.txt` report:
```python
acc = ls.accounting(data, thresholds=(50, 250, 1000, 5000, 20000), connectivity=8,
                    highlight_largest=True)
print(acc.summary())          # GTB acc.txt layout
acc.stats                     # per class: range, colour, objects, area, % objects, % FG area
acc.classes                   # uint8 GTB colour codes (see ls.ACCOUNTING_CLASSES)
acc.ids                       # int32 object IDs (0 BG, -1 missing, -3/-4 special BG)
acc.area                      # int32 object area in pixels on every object pixel
ls.write_raster(acc.classes, "forest_acc.tif")   # with GTB's colour table
```
* Input: `uint8`, 0 = missing, 1 = background, 2 = foreground, 3/4 = special background.
* `thresholds`: 1 to 5 non-decreasing upper bounds (repeats dropped). Class 1 is
  `[1, t1]`, class k is `(t(k-1), tk]`, and the last class holds the larger objects.
  `"auto"` uses the classes GTB's dialog proposes (0.1/1/10/50/90% of the 8-connected
  object-area range).
* IDs are numbered as in GTB: raster order starting from the bottom row, because GTB
  flips the image before `label_region`.
* `highlight_largest` paints the three largest objects pink (code 80), like
  GTB's "big3pink". It needs at least three objects. Objects of equal area share a rank.
* The median is the upper middle value for an even count (IDL `MEDIAN`), and the
  standard deviation is the sample one (IDL `MOMENT`).

### Isochrone Maps and Restoration Setup
GTB's *Restoration Planner*. Resistance maps are coded 2 = foreground (cost 1) and
3–100 = background resistance; marker maps 1 = object A, 2 = object B, 3 = missing.
A step between neighbouring pixels (8-connectivity, diagonals included) costs the mean
of their resistances, as in GTB's `ggeo` program:
```python
resist = ls.distance_to_resistance(forest)        # or any 2/3-100 resistance map
iso = ls.isochrone_map(resist, marker)            # cost from A
iso.map      # 0-100 scaled cost, 101 markers, 102 missing, 104 unreachable
iso.cost     # cost as GTB rounds it
ab = ls.isochrone_map_ab(resist, marker)          # cheapest A-B route through every pixel
ab.lcp       # least-cost path (1 pixel wide), code 103 in ab.map
```
The isochrone maps are identical to GTB's (tested against GTB's code and its `ggeo`
program). `cost_distance()` / `least_cost_path()` above use a different cost model
(floating point, diagonal steps weighted by √2).

### Landscape Parcellation
Computes patch division, effective mesh size, and splitting indices:
```python
parc = ls.parcellation(data, connectivity=8, return_labels=True)
print(parc.summary())
# Class-level metrics table:
# - APS (Average Patch Size)
# - AWAPS (Effective Mesh Size / Area-Weighted APS)
# - DIVISION (Landscape Division Index)
# - PARC (Logarithmic Parcellation Index [%])
```

### Habitat Contortion
Calculates boundary tortuosity and perimeter inflection points:
```python
contortion_map = ls.contortion(data, min_area=1)
# Values represent shape tortuosity per component
```

---

## 10. Change Detection & Spatial Convolution

### Simple Change (A $\to$ B)
Computes 4-class binary transition:
```python
sc = ls.simple_change(map_t1, map_t2)
print(sc.summary())
# sc.change_map values:
# 11: Stable Background
# 12: Foreground Gain
# 21: Foreground Loss
# 22: Stable Foreground
# 255: Missing / Nodata
```

### Morphological Change Detection (MCD)
Eliminates spurious boundary chatter through geodesic reconstruction of core changes ($d > 2$) and computes landscape elasticity:
```python
mcd = ls.morphological_change(map_t1, map_t2)
print(mcd.summary())
# mcd.change_map values:
# 11: Stable BG
# 12: MCD Gain (real structural expansion)
# 21: MCD Loss (real structural loss)
# 22: Stable FG
# 176: Spurious boundary chatter
# mcd.elasticity: ratio of core change to total area change
```

### Fragmentation (FOS) Change
Produces continuous delta FOS maps and a $6 \times 6$ density class transition matrix:
```python
fc = ls.fos_change(fos_t1, fos_t2)
print(fc.summary())
print(fc.matrix_6x6)
# - fc.change_map: delta FOS shifted by +100
# - fc.frag_decrease_pct: % of shifts with reduced fragmentation
# - fc.frag_increase_pct: % of shifts with increased fragmentation
```

### FAD Multiscale Change
GTB's *Change → FAD Multiscale*: for every FAD-MS observation scale and the multiscale map,
a 7×7 table of pixels moving between fragmentation classes (background, rare, patchy,
transitional, dominant, interior, intact) from time A to time B:
```python
ch = ls.fad_multiscale_change(forest_2000, forest_2020)   # maps, FAD-MS results or GTB *_fad_mscale.tif
sc = ch.scales["mscale"]            # or ch.scales[7], ch.scales[13], ...
sc.matrix                           # 7x7 pixel counts, rows A0-Background..A6-Intact, columns B0..B6
sc.percent                          # 6x6 foreground transitions in % of the changed pixels
sc.frag_decrease_pct, sc.frag_increase_pct   # above / below the matrix diagonal
ch.averages                         # FAD_AV and AVCON at A and B per scale, and their change
print(ch.summary("mscale"))         # GTB FAD_change_mscale.txt layout
```
Both maps must have the same size, and either both or neither may contain non-fragmenting
background (input 4). The tables are identical to GTB's.

### Landscape Mosaic Heatmap Change
GTB's *Change → Landscape Mosaic Heatmap*: the difference B − A of two heatmaps.
```python
d = ls.landscape_mosaic_change(lm_input_2000, lm_input_2020, window_size=27)
d.table            # percent_a, percent_b, change per LM class
d.to_csv("delta_heatmap.csv")   # identical to GTB's delta_heatmap.csv
```

### Continuous Grayscale Spatial Convolution (GraySpatCon)
Computes moving-window spatial metrics over continuous grayscale/percentage rasters ($0..100$):
```python
gsc_mean = ls.gray_spatial_convolution(img, metric="mean", window_size=7)
gsc_std = ls.gray_spatial_convolution(img, metric="stddev", window_size=7)
gsc_ent = ls.gray_spatial_convolution(img, metric="shannon_diversity", window_size=7)
```
**Supported Metrics**:
* `"mean"` / `"average"`
* `"median"`
* `"majority"` / `"mode"`
* `"min"` / `"max"` / `"range"`
* `"stddev"`
* `"shannon_diversity"`
* `"shannon_evenness"`
* `"diagonal_contagion"`

---

## 11. Batch Processing & Raster Collections

Any tool can run on many rasters (GTB's *Batch Process*):
```python
results = ls.batch("accounting", "tiles/*.tif", "out/", highlight_largest=True, max_memory="16GB")
ls.batch("spa", "tiles/", "out/", variant=6)
ls.batch(my_function, "tiles/", "out/")     # any function of one raster
[r.outputs for r in results]                # GeoTIFF maps, CSV tables, TXT report, JSON record
```
Outputs are named `<name>_<tool>[_<field>].<ext>`. The JSON record
(`<name>_<tool>.json`) holds the parameters, the output files and the scalar
results, and lets a rerun skip finished inputs (`overwrite=True` recomputes them).

Process thousands of raster tiles efficiently without exhausting system RAM:

```python
# Header-only scan of directory
tiles = ls.load_raster("path/to/tiles/")

# Batch process with bounded memory (max 16 GB RAM used concurrently)
results = ls.mspa_batch(
    tiles,
    output_dir="./mspa_output",
    edge_width=2,
    n_workers=8,
    max_memory="16GB",
    overwrite=False,  # resumes if interrupted
)
```

---

## 12. Accuracy & Bit-for-Bit Verification

`landschaft` is validated by an automated test suite containing **324 test cases**:
* **MSPA**: Tested against `mspa_win64.exe` on 1,200+ random configurations and full CORINE datasets (0 differing pixels).
* **FAD & Contagion**: Tested against original JRC `spatcon64` binaries (bit-exact).
* **Landscape Mosaic**: Tested against JRC `recode64` binaries (bit-exact).
* **FAD-MS, FAD Multiscale Change, LM Heatmap and Heatmap Change**: Tested against GTB's own IDL code run in GDL together with GTB's `spatcon` (FAD-MS maps and 103-class maps identical, change tables identical, heatmap vectors identical bit for bit, CSV reports identical character for character).
* **GraySpatCon**: Tested directly against `grayspatcon64.exe` (parity verified to float32 precision).
* **MSPA Tiling**: Tested against GTB's `mspatile` IDL code run in GDL with GTB's gdal tools and `mspa` (10 stored cases covering all tiling layouts and tile shortcuts, plus live random cases; maps identical). GDL needs two shims for GTB's file handling (`tests/gtb_idl/read_tiff.pro`, `updategeotagsinimage.pro`).
* **FOS** (FAD/FED/FAC, 5/6 classes, binary and grayscale): Tested against GTB's batch FOS IDL code run in GDL with GTB's `spatcon` and `grayspatcon` (maps identical; statistics identical to IDL's float/double precision). 13 stored cases plus live random scenes.
* **Accounting** (IDL-only in GTB): Tested against GTB's own IDL code (`tests/gtb_idl`, statements copied verbatim from `guidostoolbox.pro`) run in GDL. Maps and integer statistics are identical. Stored digests (`tests/data/reference/idl_reference.json`) cover 7 maps × 5 parameter sets, and random scenes run live when GDL is available.

To run the verification suite:
```bash
pytest -v
```

---

## 13. Landscape Metrics (FRAGSTATS / landscapemetrics)

The patch-, class- and landscape-level metrics of FRAGSTATS as implemented by the R
package [landscapemetrics](https://r-spatialecology.github.io/landscapemetrics/):
same metric names (`lsm_p_area` → `"area"` at the patch level, `lsm_c_ca` →
`"ca"`, …), same options, same patch ids and the same values: the same bits for all
metrics but `circle`, `gyrate` and rare `enn` ties, which agree to a relative
1e-10 (see *Bit-identical arithmetic* below). The FRAGSTATS statistics that
landscapemetrics lacks (`_am`, `_md`, `_ra`) are available as extras.

### How it works

`Landscape(data, ...)` maps the class values to dense indices once. On first use the
C++ core then

1. labels the patches of **all classes in one pass**: the map is cut into strips of
   64 rows labelled in parallel (union-find), and the pieces are joined across the
   strip seams;
2. accumulates, in the same pass, every per-patch statistic the metrics need: cells,
   cell sides towards other cells (left/right and top/bottom separately, for
   non-square cells), like adjacencies (orthogonal and diagonal), coordinate sums;
3. orders the patches like landscapemetrics (by class, then by first cell in
   column-major order), so patch `id`s are the same;
4. counts, in one more parallel pass, the 4-neighbour adjacencies between classes,
   NA and the landscape boundary (plus the diagonal ones when `neighbourhood=8`).

Every metric is a function of these tables, so asking for many metrics costs little
more than asking for one. The results are cached in the object.

### Usage

```python
import landschaft as ls

lc = ls.Landscape("landcover.tif")          # path, DataArray, numpy or masked array
lc                                           # Landscape(shape=..., resolution=..., classes=[...])

lc.patch_metrics(["area", "perim", "shape"])  # class, id, area, perim, shape
lc.class_metrics()                            # every class-level metric
lc.landscape_metrics(["shdi", "contag", "ed", "lsi"])
lc.metrics(level=["class", "landscape"], type="aggregation metric")   # long format
lc.metrics(what=["lsm_c_ca", "lsm_l_ta"])     # landscapemetrics function names

lc.labels()          # patch ids 1..n (0 = NA), same grid as the input
lc.to_raster("area") # patch metric painted on the patches (spatialize_lsm)

ls.landscape_metrics(["t1.tif", "t2.tif"], level="landscape")   # adds a `layer` column
ls.list_metrics(level="class", type="-core area metric")        # "-" excludes
```

Wide tables (`patch_metrics`, `class_metrics`, `landscape_metrics`) have one column
per metric; the long table (`metrics`, `landscape_metrics()`) has the columns of
`calculate_lsm`: `level`, `class`, `id`, `metric`, `value` (plus `layer`), sorted the
same way. Both are pandas DataFrames, ready for `to_csv` / `to_parquet`.

### Input

* Cell size from the raster (`res=` for numpy arrays, default 1); areas are in
  hectares and lengths in map units, as in landscapemetrics. A geographic CRS gives
  a warning (degrees).
* NA: the raster's nodata value, `nodata=`, masked cells and NaN.
* Float maps are truncated to integers (like `terra::as.int`), with a warning when
  values are not whole numbers.

### Options (landscapemetrics arguments)

| option | default | used by |
|---|---|---|
| `directions` | 8 | patch connectivity (all patch-based metrics) |
| `count_boundary` | `False` | `te`, `ed`: landscape boundary and NA count as edge |
| `consider_boundary`, `edge_depth` | `False`, 1 | core-area metrics |
| `cell_center` | `False` | `gyrate`: centroid moved to the nearest cell |
| `classes_max` | `None` | `rpr` (NA without it) |
| `neighbourhood`, `ordered`, `base` | 4, `True`, `"log2"` | `ent`, `condent`, `joinent`, `mutinf`, `relmutinf` |

`calculate_lsm` also uses `neighbourhood` for `te`, `ed`, `iji` and `contag`,
unlike the functions `lsm_l_te`, `lsm_c_iji`, `lsm_l_contag`, … (always 4
neighbours). landschaft follows the individual functions; with the default
`neighbourhood = 4` both agree.

### Available metrics

| level | metrics |
|---|---|
| patch | `area`, `perim`, `para`, `shape`, `frac`, `gyrate`, `contig`, `circle`, `enn`, `core`, `ncore`, `cai` |
| class | `ca`, `pland`, `np`, `pd`, `lpi`, `te`, `ed`, `ai`, `clumpy`, `cohesion`, `division`, `mesh`, `split`, `iji`, `lsi`, `nlsi`, `pladj`, `pafrac`, `tca`, `cpland`, `ndca`, `dcad`, and `_mn`/`_sd`/`_cv` of `area`, `para`, `shape`, `frac`, `gyrate`, `contig`, `circle`, `enn`, `core`, `cai`, `dcore` |
| landscape | `ta`, `np`, `pd`, `lpi`, `te`, `ed`, `ai`, `cohesion`, `contag`, `division`, `mesh`, `split`, `iji`, `lsi`, `pladj`, `pafrac`, `pr`, `prd`, `rpr`, `shdi`, `shei`, `sidi`, `siei`, `msidi`, `msiei`, `ent`, `condent`, `joinent`, `mutinf`, `relmutinf`, `tca`, `ndca`, `dcad`, and the same `_mn`/`_sd`/`_cv` |

`_sd` is the sample standard deviation and `_cv` = `_sd / _mn · 100`, as in
landscapemetrics (NA for a single patch).

`enn` is the distance between the centres of the nearest cells of the patch and of
another patch of the same class; it is NA when the class has a single patch, and
so are the `enn_mn/sd/cv` of that class and of the landscape (NA propagates, as
in landscapemetrics). When every class has a single patch, landscapemetrics
returns one class-level row without class for each `enn_*`; `metrics()` does the
same, while `class_metrics()` keeps one NA row per class. The engine sorts the
patch edge cells of each class in Morton order, builds one quadtree-like tree per
class whose nodes know when all their cells belong to one patch, and searches the
nearest cell of another patch leaf by leaf, in parallel; a 5×5 raster look-up
and the smallest possible distance between two patches settle most cells without
a search. The distances are computed as in landscapemetrics and are identical.

`circle` is `1 - area / area of the smallest circumscribing circle`. As in
landscapemetrics the circle encloses the corners of the patch cells in its first
and last rows and columns, found with Welzl's algorithm. landscapemetrics shuffles
the points at random, so its values can change in the last bit between runs;
landschaft uses a fixed shuffle per patch (repeatable results, same values within
1e-15). landscapemetrics refuses non-square cells; landschaft uses the real cell
width and height.

Core areas follow landscapemetrics' `get_boundaries`: a cell is core when its
city-block distance to the nearest cell of another class or NA (and to the map edge,
unless `consider_boundary=True`) exceeds `edge_depth`; `edge_depth=0` behaves like 1.
The core engine computes that distance in two parallel separable passes instead of
repeated erosions. As in landscapemetrics, `dcore_mn/sd/cv` summarise the *number* of
core areas per patch (`ncore`), not their sizes.

### FRAGSTATS extras, metric names and threads

```python
lc.class_metrics(["area_am", "gyrate_am", "shape_md", "enn_ra"])   # FRAGSTATS statistics
lc.metrics(level="class", metric="shape", extras=True)              # shape_mn ... shape_ra
ls.list_metrics(extras=True)                                        # 211 metrics ("extra" column)
lc.class_metrics(["CA", "AREA_MN", "total_area", "shape_index_am"]) # FRAGSTATS / pylandstats names
lc.landscape_metrics(["SHDI", "shannon_diversity_index", "contagion"])
ls.Landscape("landcover.tif", n_threads=4)                          # limit the cores
```

**Extras.** For every patch metric with summary statistics (`area`, `cai`, `circle`,
`contig`, `core`, `dcore`, `enn`, `frac`, `gyrate`, `para`, `shape`) the class and
landscape levels also have the area-weighted mean `_am` (Σ x·a / Σ a, as R's
`weighted.mean`), the median `_md` and the range `_ra` of FRAGSTATS and pylandstats, and
`perim` gets `_mn`, `_sd`, `_cv`, `_am`, `_md`, `_ra` (pylandstats). `gyrate_am` is
FRAGSTATS' correlation length. `dcore` follows landscapemetrics (number of core areas per
patch), weighted by patch area. As for `_mn`, NA propagates (pylandstats skips NA at the
landscape level). These 78 metrics are not part of landscapemetrics, so they are only
selected with `extras=True` (or by name, or in `what`); without it every selection gives
the same rows as `calculate_lsm`. They were checked against numpy and against
pylandstats' `_mn`/`_am`/`_md`/`_ra` of `area`, `perimeter`, `fractal_dimension` (and
`euclidean_nearest_neighbor`, `core_area` where pylandstats' patch values agree).

**Names.** Wide tables and `to_raster` accept FRAGSTATS abbreviations in capitals
(`"SHDI"`, `"AREA_MN"`, `"ENN_AM"`), pylandstats' method names (`"total_area"` → `ca` at
the class level and `ta` at the landscape level, `"proportion_of_landscape"`,
`"shape_index_am"`, `"disjunct_core_area_mn"`, `"continguity_index"`, …) and FRAGSTATS'
full names (`"shannon_diversity_index"`, `"effective_mesh_size"`, …); the columns keep
the names as asked. The `metric=` filter accepts family aliases (`"shape_index"`).

**Threads.** `n_threads` (in `Landscape`, `landscape_metrics`, `sample_metrics`,
`window_metrics`) limits the C++ core; the results do not depend on it.

**FRAGSTATS-only metrics** (extras; landscapemetrics has none of them):

```python
contrast = {(1, 2): 0.2, (1, 3): 0.7, (2, 3): 0.5}        # or a class x class DataFrame
lc = ls.Landscape("landcover.tif", search_radius=500, connect_threshold=100,
                  contrast=contrast, boundary_contrast=0.0, similarity={(1, 2): 0.5})
lc.patch_metrics(["prox", "simi", "econ"])
lc.class_metrics(["connect", "cwed", "teci", "prox_mn", "econ_am"])
```

| metric | level | definition |
|---|---|---|
| `prox` | patch (+ stats) | Σ over the other patches of the class within `search_radius`: area (m²) / distance² |
| `simi` | patch (+ stats) | the same over the patches of all classes, weighted by the `similarity` of the classes (1 within a class; without a table `simi = prox`) |
| `connect` | class, landscape | % of the pairs of patches of a class within `connect_threshold` |
| `econ` | patch (+ stats) | Σ edge length × contrast / perimeter × 100 (edges to NA and the map edge weigh `boundary_contrast`) |
| `cwed` | class, landscape | Σ edge length × contrast / total area (m/ha); NA and map-edge edges count with `count_boundary` |
| `teci` | class, landscape | Σ edge length × contrast / Σ edge length × 100, all edges (FRAGSTATS' E*) |

Distances between patches are the shortest distances between their cell centres, as
for `enn` and in FRAGSTATS; the neighbour's whole area counts in `prox`/`simi`. Without
`search_radius`, `connect_threshold` or a `contrast` table the metrics are NaN; contrast
pairs missing from the table give NaN too. The pairs of patches within reach come from a
range search over the boundary cells, one focal patch at a time (2000×2000 with 223,000
patches: `prox` 0.2 s with a 3-cell radius, 2 s with 30 cells). These metrics need the
raster, except `cwed` and `teci`, which also work out of core. There is no FRAGSTATS to
compare with here: they are tested against direct implementations of the definitions.

**`convention="fragstats"`** changes five landscapemetrics conventions, all checked
against pylandstats or brute force:

- with non-square cells the sides between left/right neighbours are as long as the cell
  *height* (the physical perimeter; landscapemetrics multiplies them by the width), for
  `perim`, `te`, `ed` and everything built on them;
- `dcore_*` summarise the sizes (ha) of the disjunct core areas of the class or landscape
  (`dcore_am` weighted by those sizes), not the number of core areas per patch;
- `shape` is FRAGSTATS' raster SHAPE, the perimeter over the smallest perimeter of as many
  cells (in cell sides), instead of .25 p / sqrt(a); cells that are not square keep
  .25 p / sqrt(a), and `lsi` becomes .25 E / sqrt(A) with the physical edge lengths
  (pylandstats; FRAGSTATS needs square cells);
- `_sd` and `_cv` use the standard deviation over n, FRAGSTATS' (landscapemetrics uses
  R's `sd()`, over n - 1; one patch gives 0 instead of NA);
- the `circle` encloses the whole patch: landscapemetrics' circle only encloses the
  corners of the cells in the first and last rows and columns, which can miss the
  corners that decide it. Here the first and last cell of the patch in every row are
  gathered in parallel, reduced to their convex hull on the integer grid (exact), and
  Welzl's algorithm runs on the hull (also out of core, with the same values;
  `circumscribing_circles` follows the convention too).

### Bit-identical arithmetic

R computes `sum()`, `mean()` and `var()` in `long double`, the 80-bit x87 type with a
64-bit mantissa (R on Windows is built with gcc/mingw-w64). MSVC has no such type, so
`src/cpp/lsm/extended.hpp` emulates it with integer arithmetic, correctly rounded (half to
even); it was checked against gcc's native type on 40 million random operations, and
gcc/clang builds use the native type. The metrics then follow landscapemetrics' R code
operation by operation: `mean()` sums, divides and adds the mean residual; `sd()` squares
the deviations from the mean rounded to double; `sum()` inside `aggregate()` runs per
class in patch order; `weighted.mean`, `median` and the range are computed the same way
for the extras.

R's `log()` and the `sqrt()` of a vector come from mingw-w64's math library, which runs
them on the x87 unit (glibc's `fyl2x`/`fyl2xp1` code for `log`) and rounds twice, so they
can differ in the last bit from a correctly rounded result; `rmath.hpp` does the same
(with a small MASM routine under MSVC, which has no x64 inline assembly). R's byte-code
interpreter computes the `sqrt` of a single number inline and correctly rounded (the one
inside `sd()`, `cohesion`'s factor, the shape of a single patch), and `x^-1` is `powl`.
`pafrac` is 2 over the slope of `lm(log(area) ~ log(perimeter))`: `src/cpp/lsm/rlm.cpp`
runs what `lm()` runs, LINPACK's `dqrdc2` (Householder QR with R's limited pivoting,
tolerance 1e-7) and `dqrsl` over R's reference BLAS (`ddot`, `daxpy` and `dscal` unrolled
as in R, LAPACK 3.12's `dnrm2`), so the slope is R's to the last bit and NA exactly when R
drops the perimeter as collinear.

Values that stay within 1e-10 instead of identical: `circle` (landscapemetrics shuffles
Welzl's points at random, so its own values change between runs), `gyrate` (R sums the
coordinates and distances cell by cell in extended precision) and the `enn` of rasters
whose cell sides differ by rounding only (R's pruned search can settle on another pair of
cells at the same distance).

landscapemetrics' own quirks are reproduced: patch perimeters treat cell sides that
differ by rounding only as equal (`all.equal`), and `window_metrics` computes `shdi` and
`shei` from 8-connected patch areas whatever `directions` is, as `window_lsm` does (their
functions hard-code it; `calculate_lsm` hides it).

### Rasters larger than memory (`block_rows`)

```python
lc = ls.Landscape("huge_landcover.tif", block_rows="auto")   # or a number of rows
lc.metrics(level=["class", "landscape"])
ls.landscape_metrics("huge_landcover.tif", block_rows=2048, what=["lsm_c_ca", "lsm_l_shdi"])
```

With `block_rows` the file is read by blocks of rows (rounded up to a multiple of 64;
`"auto"`: 64 rows per thread, at least 4 and at most 32 million cells) and only one
block, the patch tables and the class adjacencies are in memory. The results are
**identical to the in-memory ones** (all 133 metrics, every option), so they match
landscapemetrics as well.

1. Pass 0 finds the class values (a count of the raw values for 8/16-bit files).
2. Pass 1 labels every block with the in-memory engine. The pieces of the patches cut by
   the block seams are joined with a union-find, and what the block labelling got wrong
   at the seams is corrected: cell sides towards the next block, cell pairs across the
   seam and, with 4 directions, diagonal pairs whose pieces turn out to be one patch.
   Class adjacencies are counted with one row of context above and below. This gives
   the patch tables (same patches, same ids) and every metric but the next four.
3. A detail pass, run only when they are asked for, computes core areas (distance to
   the patch edges with `edge_depth` rows of context; core areas joined across seams
   like the patches), `gyrate` (plus a pass for the ties with `cell_center`), `circle`
   (cells on the edges of each patch's bounding box) and `enn`.

Blocks start at multiples of 64 rows, the height of the strips the in-memory engine
labels in parallel, so even the floating-point sums (`gyrate`) are made in the same
order. Memory grows with the number of patches (about 70 bytes per patch piece while
reading, plus the per-patch values the metrics keep) and, for `enn` only, with the
number of patch edge cells of the classes with several patches (about 24 bytes each):
enn needs every edge cell to find nearest neighbours. `labels()` and `to_raster()` are
not available out of core. GeoTIFF blocks are decoded on all cores
(`GDAL_NUM_THREADS`).

### Sample plots, moving windows and several rasters

```python
# metrics of sample plots (sample_lsm): squares or circles around points, or polygons
ls.sample_metrics("landcover.tif", points_xy, size=500, shape="circle", level="landscape")
ls.sample_metrics("landcover.tif", gdf_polygons, plot_id=gdf_polygons["name"], all_classes=True)

# landscape-level metrics in a moving window (window_lsm): {metric: raster}
maps = ls.window_metrics("landcover.tif", window=21, what=["lsm_l_shdi", "lsm_l_ed"])
disk = np.hypot(*np.mgrid[-5:6, -5:6]) <= 5          # any odd-sized shape
maps = ls.window_metrics("landcover.tif", window=disk, metric="contag")

# several rasters, computed side by side
ls.landscape_metrics(["t1.tif", "t2.tif", "t3.tif"])
```

**`sample_metrics`** cuts every plot out of the raster as `terra::crop(mask = TRUE)`
does in `sample_lsm`: the plot's bounding box is snapped to the nearest cell edges
(one cell if it collapses), every cell the plot touches is kept (GDAL's
all-touched rasterization, ported to C++ with GDAL's rules and epsilons, so no GDAL
call per plot) and the others become NA. Point buffers are
landscapemetrics' squares (half side `size`) or 100-vertex circles (radius `size`).
Each plot is then a landscape of its own; all plots go to the C++ core at once and are
computed in parallel. The table has `calculate_lsm`'s columns plus `plot_id` and
`percentage_inside` (area of the plot's cells over the plot area, in %; the plot area
is geodesic when `y` has a CRS, like a GeoDataFrame, and planar otherwise, as
`terra::expanse(transform = TRUE)` does), ordered like `sample_lsm`. `all_classes`
adds NA rows for the classes of the whole raster missing in a plot, including a row
for class NA when the raster has NA cells, as landscapemetrics does. A plot outside
the raster gives NA metrics and 0 % (landscapemetrics stops with an error).

**`window_metrics`** computes landscape-level metrics for the window centred on every
cell (cells outside the raster or outside the window shape are NA), in parallel over
the rows; each window is a small landscape with the class values mapped to dense
indices once for the whole raster. `terra::focal` hands the window values row by
row but landscapemetrics fills its matrix column by column, so its windows are
transposed; landschaft transposes them too, with the cell width and height swapped,
so the patches come in R's order (which `pafrac`'s regression rounds by) and the
values are R's. Where `window_lsm` breaks the geometry landschaft keeps it:
rectangular windows (R reshapes them), cells that are not square (R keeps the cell
sides unswapped) and window shapes that are not symmetric (R masks the transposed
values with the untransposed shape). And
`window_lsm` fails for `gyrate_*` (it does not pass `cell_center`), which work here.

**`scale_sample`** runs `sample_metrics` for several buffer sizes (a `size` column),
ordered like landscapemetrics' `scale_sample` (plot, level, metric, class, id, size);
points only.

**Zonal results and export.** `zonal_metrics(data, zones)` cuts the polygons out like
`sample_metrics` and returns a GeoDataFrame: the zones' geometry and attributes, one
column per landscape metric and one per class metric and class (`ca_11`, `pland_42`...),
and `percentage_inside`. `write_table(table, path)` writes any metric table by extension:
`.csv`, `.csv.gz`, `.parquet` (pyarrow; keeps the categorical and nullable columns,
compact for patch tables with millions of rows) or, for GeoDataFrames, `.gpkg`,
`.geojson`, `.shp`, `.fgb`. `Landscape.to_raster(metric)` paints a patch metric on the
patches or a class metric on the cells of each class, several at once with a list, and
writes GeoTIFFs with `path=`. `batch("landscape_metrics", inputs, out, **options)` writes
one long table per raster (`<name>_landscape_metrics.csv`, or `.parquet` with
`table_format="parquet"`) plus the JSON record; with `block_rows` the files are read by
blocks; `batch("window_metrics", ...)` writes one GeoTIFF per metric.

**Several rasters** (`landscape_metrics(list)`, and lists in `sample_metrics` /
`window_metrics`) are computed side by side when they are small and one after the
other, each in parallel, when they are large.

### Time series and spatial signatures (pylandstats-style)

```python
sta = ls.SpatioTemporalAnalysis(["lc2000.tif", "lc2010.tif", "lc2020.tif"], dates=[2000, 2010, 2020])
sta.landscape_metrics(["shdi", "ed", "np"])        # one row per date
sta.class_metrics(["pland", "np"])                  # (class, date) rows; absent class: 0 for area/count
sta.plot_metric("pland", classes="all")             # one line per class
ls.SpatioTemporalAnalysis(files, dates, zones=gdf)                   # per polygon: (zone, date) rows
ls.SpatioTemporalAnalysis(files, dates, points=xy, size=500)         # buffers around points
ls.SpatioTemporalAnalysis(files, dates, grid=1000)                   # 1 km grid (ls.grid_zones)

sig = ls.SpatialSignatureAnalysis(files, landscape_metrics=["shdi", "ed"], class_metrics=["pland"])
sig.table                     # the signature matrix: rows = landscapes (or zones), columns = metrics
res = sig.pca()               # scores, loadings, explained_variance_ratio
labels = sig.cluster(3)       # k-means (k-means++, best of 10)
sig.plot_pca(labels); sig.plot_cluster_metrics(labels)
ls.SpatialSignatureAnalysis("map.tif", grid=1000, landscape_metrics=[...])   # zones of one map
```

The series are computed with `landscape_metrics` / `sample_metrics` (so they are the same
values). The signature matrix has one column per landscape metric and one per class
metric and class (`pland_11`); the PCA works on z-scores (population standard deviation,
as scikit-learn's `StandardScaler`), leaves out constant metrics and rows with NaN (with a
warning), and orients each component so that its largest loading is positive; k-means runs
on the z-scores or on the first principal components. Both agree with scikit-learn, which
is not needed.

### Plots

With matplotlib (`pip install landschaft[plot]`), like landscapemetrics' `show_*` (they
return a `matplotlib.figure.Figure`; the maps use the raster's coordinates):

```python
lc.show_patches()                     # all patches, each in its own colour
lc.show_patches("all", labels=True)   # one map per class, with patch ids
lc.show_cores(edge_depth=2)           # core (dark) and edge (light) cells per class
lc.show_lsm("shape", "all")           # a patch metric on the patches, with a colour bar
ls.show_correlation(lc.metrics(level=["patch", "class"]), method="spearman", labels=True)
ls.show_patches("landcover.tif", directions=4)    # rasters work too (Landscape options)
```

`classes` is `"global"` (one map), `"all"` (one per class) or class values;
`show_cores` defaults to `consider_boundary=True` as in landscapemetrics' plot.

### Utilities (landscapemetrics' `get_*`, `extract_lsm`, `check_landscape`, `calculate_correlation`)

```python
lc = ls.Landscape("landcover.tif")
lc.centroids()                          # class, id, x, y (get_centroids)
lc.centroids(cell_center=True)          # the patch cell nearest to it (ties: several rows)
lc.nearest_neighbours(return_id=True)   # class, id, dist, id_neighbour (get_nearestneighbour)
lc.circumscribing_circles("class")      # diameter and centre (get_circumscribingcircle)
lc.adjacencies(8, what="unlike")        # class x class cell pairs (get_adjacencies)
lc.boundaries(edge_depth=2, patch_id=True)   # get_boundaries of the patches
lc.patches()                            # {class: raster of patch ids} (get_patches)
lc.extract_metrics(points_xy, what=["lsm_p_area", "lsm_p_shape"])   # extract_lsm
ls.raster_boundaries("patch_ids.tif")   # get_boundaries on the values of any raster
ls.check_landscape(["a.tif", "b.tif"])  # crs, units, class, n_classes, OK
ls.calculate_correlation(ls.landscape_metrics(["t1.tif", "t2.tif", "t3.tif"]), method="spearman")
```

All give landscapemetrics' values, bit for bit (except the circles, randomised in
landscapemetrics), and follow its conventions:

* **Coordinates** are terra's: cell centres `xmin + (col + 0.5) · (xmax − xmin) / ncol`,
  the extent from the geotransform (`xmax = xmin + res · ncol`). A point on a cell edge
  belongs to the cell right of / below it, except on the last column / row (terra's
  `cellFromXY`); points outside or on NA give no rows in `extract_metrics`.
* **Centroids** are R's `mean()` of the cell centres in raster order (80-bit sums). With
  `cell_center`, the nearest cells are those at the smallest distance (x87 `sqrt`, as R
  computes a vector); ties come in the order R's `merge()` leaves them (its Shell sort).
* **Nearest neighbours**: the distance of each patch to the nearest patch of its class is
  the `enn` metric. With `return_id`, landscapemetrics gives each boundary cell (4
  neighbours) the first cell of another patch at its nearest distance that its sweep over
  the cells sorted by x meets; the patch lists the patches its cells meet at the patch's
  distance. landschaft applies the same rule to the candidates of a range search around
  each cell, so it does not slow down on long columns of cells.
* **Circles** enclose the corners of the cells in the first and last rows and columns
  of the patch (or class), as the `circle` metric; centres in map coordinates. Non-square
  cells work (landscapemetrics refuses them).
* **Adjacencies** count directed pairs (each pair twice) with 4, 8 or any neighbourhood
  matrix (1 = neighbour, 0 = focal cell; read transposed, as landscapemetrics does); NA
  cells are left out. `what="like"/"unlike"/"triangle"` keep the diagonal / the lower (or
  `upper`) triangle, NaN elsewhere.
* **Boundaries**: `Landscape.boundaries` marks, for every class separately, the cells
  within `edge_depth` cells (city-block) of another class, NA or the map edge (unless
  `consider_boundary`): the complement of the core cells. `raster_boundaries` runs
  landscapemetrics' `rcpp_get_boundaries` rule on the values as given: a cell is a
  boundary when the cell *below* differs or the cell above, left or right is NA, so on a
  multi-class raster only one side of a change is marked, as in R; `patch_id` is refused
  with a warning when a value is 0.
* **check_landscape** reads the CRS like terra, which assumes lon/lat (`OGC:CRS84`) for a
  raster file without CRS whose extent fits longitudes and latitudes.
* **calculate_correlation** builds the tables like R's `xtabs` (values summed per id /
  class / layer, missing combinations 0, NA values NA, rows without id left out) and
  computes R's `cor()`: two-pass means and 80-bit cross products, Spearman on average
  ranks, Kendall from sign products; NA or constant metrics have NA correlations (left
  out) and a diagonal of 1.

These need the raster, so they are not available out of core (`block_rows`).

### Verification

`tools/make_reference_lsm.py` runs landscapemetrics (`tools/lsm_reference.R`) on 14
maps (its `landscape`, crops of `augusta_nlcd` and `podlasie_ccilc`, synthetic maps
with NA cells, negative and large class values, non-square cells, salt-and-pepper
patterns, few distant patches with a non-integer cell size, a checkerboard, single
rows and columns, a single class) with 5 option sets, and stores every value in `tests/data/reference/lsm`. `tests/test_lsm_reference.py`
compares all of them; with `Rscript` available (`LANDSCHAFT_RSCRIPT`,
`LANDSCHAFT_R_LIBS` for a specific landscapemetrics library) it also runs
landscapemetrics live on random maps with other option combinations.
`make_reference_lsm.py sample window` also stores `sample_lsm` (square, circle and
polygon plots, with and without `all_classes`, a CRS and boundary options) and
`window_lsm` (square and circular windows) results, which `tests/test_lsm_sampling.py`
compares, plus live runs on random maps; sample plots and windows are also checked
against `Landscape` objects built on the same cells. `tests/test_lsm_tiled.py` checks
that out-of-core results (`block_rows`) are identical to the in-memory ones on every
stored reference map and on files with many blocks, patches across every seam, deep
`edge_depth` context, several data types and nodata values.
All comparisons with landscapemetrics require the same bits, except for the three
families above (relative 1e-10); `tests/test_lsm_rmath.py` checks the arithmetic itself
against R on 600 stored vectors and the `lm()` slope on 400 regressions, 166 of them
collinear (`make_reference_lsm.py rmath`).
`tests/test_lsm_extras.py` checks the extras against numpy and pylandstats, the names
and the threads. `make_reference_lsm.py utils` stores the results of the utilities
(`tools/lsm_utils_reference.R`) on 8 maps and option sets, compared by
`tests/test_lsm_utils.py`, which also runs them live on random maps. `tests/test_lsm.py` checks values computed by hand, the labels against SciPy, the
parallel seams, input handling and metric selection; `enn` against SciPy's exact
Euclidean distance transform and `circle` against an exhaustive search over the
circles through two or three hull points.

### Performance

| map | landscapemetrics 2.2.2 | landschaft |
|---|---|---|
| `augusta_nlcd`, 440×678, the 91 metrics of part 1 | 8.6 s | 0.1 s |
| `augusta_nlcd`, all 133 metrics | 15.6 s | 0.03 s |
| `augusta_nlcd`, `enn` / `circle` | 2.4 s / 0.14 s | 0.014 s / 0.008 s |
| 10,000×10,000, 10 classes, 7.7 M patches: labelling | – | 1.4 s |
| same, all class and landscape metrics | – | 2.2 s more |
| same, core areas (distance, core cells, core labelling) | – | 1.0 s more |
| same, `enn` / `circle` | – | 2.0 s / 0.5 s more |
| `augusta_nlcd`, 6 circular plots, all metrics (`sample_lsm`) | 3.7 s | 0.02 s |
| 4000×4000, 10,000 plots, all metrics (10 M values), squares / circles | – | 2.2 s / 2.2 s |
| 9×11 map, 5×5 window, 10 metrics (`window_lsm`) | ~4 s | 0.003 s |
| 1000×1000, 21×21 window, all 66 landscape metrics | – | 9 s |
| 20,000×20,000 file (400 M cells, 334 k patches), class and landscape metrics without core/gyrate/circle/enn: in memory / `block_rows="auto"` | – | 3.4 s, 3.7 GB / 3.9 s, 0.8 GB |
| same, all 133 metrics: in memory / `block_rows="auto"` | – | 8.6 s, 8.7 GB / 13.1 s, 3.3 GB |

