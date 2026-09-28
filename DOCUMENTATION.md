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
14. [Habitat Connectivity Graphs (Conefor Sensinode / Makurhini)](#14-habitat-connectivity-graphs-conefor-sensinode--makurhini)
15. [Circuit Connectivity (Circuitscape / Omniscape)](#15-circuit-connectivity-circuitscape--omniscape)
16. [Neutral Landscape Models (NLMR)](#16-neutral-landscape-models-nlmr)

---

## 1. Architecture & Design Principles

* **Line-by-line Mathematical & Logical Parity**: Reimplemented directly from GuidosToolbox C, IDL, and miallib sources (`fsp.c`, `spatcon.c`, `GraySpatCon.c`, `recode.c`, `guidostoolbox.pro`).
* **Maximum Performance via C++ and OpenMP**: Heavy raster operations (sliding windows, geodesic reconstructions, topological graph searches, Euclidean distance transforms) run in C++17 with multi-threaded OpenMP execution and GIL release.
* **$O(1)$ Sliding Window Algorithms**:
  * Forest Area Density (FAD) uses **2D Integral Images** (Summed-Area Tables) for $O(1)$ computation per pixel regardless of window size ($7 \times 7$ up to $243 \times 243$).
  * GraySpatCon uses **101-bin sliding histograms** that update columns in $O(1)$ rather than re-scanning the window.
* **Full Geospatial Metadata Preservation**: All high-level functions accept and return `xarray.DataArray` objects, seamlessly preserving CRS, affine geotransforms, spatial coordinates, and nodata masks.
* **Threads: all cores but one by default.** Every function that runs in parallel takes `n_threads` (and the ones that run several rasters at once, `n_workers`); `None` or 0 means all the cores the process may use but one, so the machine stays responsive. `ls.default_threads()` gives that count, `ls.set_default_threads(n)` changes it for the whole process (`None`: back to all cores but one). The compiled code follows the same rule even where a function has no thread argument of its own (every call into it sets the OpenMP thread count of the calling thread), in any Python thread. Results never depend on the number of threads.

```python
ls.default_threads()               # e.g. 19 on a 20-core machine
ls.fad(data, n_threads=4)          # this call: 4 threads
ls.set_default_threads(8)          # every call from now on: 8 threads
```

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
    n_threads=None,   # OpenMP threads (None / 0: all cores but one)
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

Conefor's own indices (NC, NL, H, CCP, LCP, IIC, F, AWF, PC) and node importances on
any table of nodes and distances are in [section 14](#14-habitat-connectivity-graphs-conefor-sensinode--makurhini).

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
`window_metrics`, ...) limits the C++ core (default: all cores but one); the results do
not depend on it.

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

How each platform gets R's bits (all checked by the same tests):

| Build | 80-bit sums (`sum`, `mean`, `var`) | `log`, vector `sqrt` |
| --- | --- | --- |
| Windows, MSVC x64 (the wheels) | x87 unit, MASM (`x87_msvc.asm`) | x87 unit, MASM |
| Linux / macOS on x86-64 (gcc, clang) | native `long double` | x87 inline assembly |
| Windows, MinGW gcc | emulated (`SoftExtended`): Windows runs the x87 at 53 bits | x87 inline assembly |
| ARM64 (Apple M1–M6, Linux aarch64) | emulated (`SoftExtended`): the same bits | the system library: may differ in the last bit |

On ARM64 the sums still give R's bits (R's reference values come from R on x86-64
Windows); only metrics that take a logarithm or a square root may differ in the last bit
there. The MASM file is only assembled by MSVC on x86-64; every other build uses the C++
code.

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
not available out of core. GeoTIFF blocks are decoded on the same threads
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

**`full_name=True`** (in `landscape_metrics`, `Landscape.metrics`, `sample_metrics`,
`scale_sample` and `extract_metrics`) adds `calculate_lsm`'s columns `name`, `type`
and `function_name` after `value` (categoricals; the FRAGSTATS extras have their own).
The names are landscapemetrics' own, typo included (`siei` is the "simspon's evenness
index"; `list_metrics(name=...)` takes both spellings). landscapemetrics'
`sample_lsm(full_name = TRUE)` loses `function_name` by dropping a column by position;
it is kept here.

**`return_raster=True`** in `sample_metrics` adds `raster_sample_plots`: each plot's
raster as `terra::crop(mask = TRUE)` returns it (a DataArray on the snapped window,
NaN outside the plot and on NA cells, the raster's CRS; `None` for a plot outside the
raster). The window's extent is terra's (`align` through terra's origin, then
`setWindow`), the same bits as a crop of the intact raster. terra's crop also rewrites
the extent of the raster it crops (its `removeWindow` rebuilds it with the window's
cell size), so in `sample_lsm` the plots after the first see an extent moved by a few
ulps; that side effect is not reproduced (the cells and values are the same).

landscapemetrics' public helpers:

```python
ls.construct_buffer(points_xy, "circle", 500)                    # GeoDataFrame of plots
ls.construct_buffer(points_xy, "square", 500, return_vec=False)  # R's vertex matrix
ls.pad_raster("landcover.tif", -999, 2)   # 2 rows / columns of -999 around it
ls.unpad_raster(padded, 2)                # and back
ls.raster_to_points("landcover.tif", return_NA=False)   # layer, x, y, z
ls.get_unique_values(["a.tif", "b.tif"])  # {"layer_1": values, "layer_2": values}
```

`construct_buffer` gives the plots `sample_metrics` builds (squares of half side
`size`, 100-vertex circles), with R's bits: the circle's `sin`/`cos` are the x87
`fsin`/`fcos` R runs through mingw-w64 (`np.sin` differs in the last bit at 5 of the 100
angles). `pad_raster` / `unpad_raster` return a DataArray for georeferenced input
(extent grown or shrunk by the cells, as `ext(x) + res(x) * n`; landscapemetrics fails
when the cells are not square, since terra recycles `c(xres, yres)` over the four
sides; landschaft grows each axis by its own cell size) and a numpy array otherwise;
`global_` is R's `global` (NA cells take the pad value too). `raster_to_points` gives
terra's cell centres and the values in row order. `get_unique_values` returns the sorted
distinct values of each layer (NA left out, doubles truncated), in parallel in C++.

**Zonal results and export.** `zonal_metrics(data, zones)` cuts the polygons out like
`sample_metrics` and returns a GeoDataFrame: the zones' geometry and attributes, one
column per landscape metric and one per class metric and class (`ca_11`, `pland_42`...),
and `percentage_inside`. `write_table(table, path)` writes any metric table by extension:
`.csv`, `.csv.gz`, `.parquet` (pyarrow; keeps the categorical and nullable columns,
compact for patch tables with millions of rows) or, for GeoDataFrames, `.gpkg`,
`.geojson`, `.shp`, `.fgb`. For maps with tens of millions of patches,
`Landscape.write_metrics(path, ...)` writes the table of `metrics(...)` (same rows and
types) one metric at a time and the patch metrics `chunk_rows` rows at a time, and
`Landscape.write_patch_metrics(path, metrics)` the wide patch table the same way, to
`.parquet` (one row group per chunk), `.csv` or `.csv.gz`, never building the whole
table: the 9 patch metrics of 9 M patches (81 M rows) go to Parquet in 8.0 s with
1.0 GB of extra memory, instead of 10.3 s and 6.1 GB through a DataFrame. The CSV
text is the same as `write_table`'s. `Landscape.to_raster(metric)` paints a patch metric on the
patches or a class metric on the cells of each class, several at once with a list, and
writes GeoTIFFs with `path=`. `batch("landscape_metrics", inputs, out, **options)` writes
one long table per raster (`<name>_landscape_metrics.csv`, or `.parquet` with
`table_format="parquet"`, streamed by `write_metrics`) plus the JSON record; with
`block_rows` the files are read by
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
sig.cluster(3, method="ward") # hierarchical: ward, average, complete, single, centroid, median, weighted
nmf = sig.nmf(3)              # non-negative factors: scores (W), components (H), reconstruction_error
cg = sig.clustergram(range(1, 9), method="ward")   # labels and cluster means for each k
sig.plot_clustergram(cg)      # Schonlau's clustergram, as pylandstats' get_cgram draws it
sig.plot_pca(labels); sig.plot_cluster_metrics(labels)
ls.SpatialSignatureAnalysis("map.tif", grid=1000, landscape_metrics=[...])   # zones of one map
```

The series are computed with `landscape_metrics` / `sample_metrics` (so they are the same
values). The signature matrix has one column per landscape metric and one per class
metric and class (`pland_11`); the PCA works on z-scores (population standard deviation,
as scikit-learn's `StandardScaler`), leaves out constant metrics and rows with NaN (with a
warning), and orients each component so that its largest loading is positive; k-means and
the hierarchical linkages (scipy, Euclidean, the tree cut into k clusters) run on the
z-scores or on the first principal components, and clusters are numbered by first
appearance. `nmf` scales each metric to [0, 1] (min-max) and factorises it into two
non-negative matrices with scikit-learn's method (NNDSVDa start, coordinate-descent/HALS
sweeps, Frobenius loss): the same loss as `sklearn.decomposition.NMF(init="nndsvda",
solver="cd")` to the stopping tolerance. `clustergram` gives, for each k, every row's
cluster and the mean of that cluster: its centre projected on the first principal axis of
the clustered data (`pca_weighted=True`) or averaged over the columns, the values the
clustergram package draws (pylandstats' `get_cgram`); `plot_clustergram` draws them the
same way. All agree with scikit-learn / clustergram, which are not needed.

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
| 10,000×10,000: preparing the map (class values, NA, per-class counts) | – | 0.07 s |
| 10,000×10,000, 10 classes, 7.7 M patches: labelling | – | 1.4 s |
| same, all class and landscape metrics | – | 2.2 s more |
| same, core areas (distance, core cells, core labelling) | – | 1.0 s more |
| same, `enn` / `circle` | – | 2.0 s / 0.5 s more |
| `augusta_nlcd`, 6 circular plots, all metrics (`sample_lsm`) | 3.7 s | 0.02 s |
| 4000×4000, 10,000 plots, all metrics (10 M values), squares / circles | – | 2.2 s / 2.2 s |
| 9×11 map, 5×5 window, 10 metrics (`window_lsm`) | ~4 s | 0.003 s |
| 1000×1000, 21×21 window, all 66 landscape metrics | – | 8.7 s |
| 20,000×20,000 file (400 M cells, 334 k patches), class and landscape metrics without core/gyrate/circle/enn: in memory / `block_rows="auto"` | – | 3.4 s, 3.7 GB / 3.9 s, 0.8 GB |
| same, all 133 metrics: in memory / `block_rows="auto"` | – | 8.6 s, 8.7 GB / 13.1 s, 3.3 GB |


---

## 14. Habitat Connectivity Graphs (Conefor Sensinode / Makurhini)

The graph-based connectivity indices of Saura and colleagues, as computed by
[Conefor Sensinode](http://www.conefor.org) and the R package
[Makurhini](https://github.com/connectscape/Makurhini): habitat patches are the
nodes (with an attribute, usually their area), and the distances (or dispersal
probabilities, or links) between them are the connections.

```python
import landschaft as ls

# nodes: attribute per patch; connections: n × n distance matrix or (from, to, distance) table
g = ls.HabitatGraph(areas, distances, ids=patch_ids, landscape_area=total_area)

g.overall(distance=[5_000, 10_000], probability=0.5)   # one row per distance
g.node_importance(["PC", "IIC"], distance=10_000)      # dPC, dPCintra, dPCflux, dPCconnector, dIIC...
g.components(distance=10_000)                          # component of every node
g.link_importance("PC", distance=10_000)               # dPC_removal of every link (MK_dPCIIC_links)
g.link_change(new_distances, "PC", distance=10_000)    # dPC_change of pairs with new distances
g.node_importance(["BC", "BCIIC", "BCPC"], distance=10_000)  # Conefor's betweenness centralities
g.centrality(10_000)                                   # degree, eigen, close, BWC, cluster (MK_RMCentrality)
g.centrality(10_000, probability=0.5)                  # weighted by p (strength, 1/p as lengths)

# ECA over time (Makurhini's MK_dECA): one graph per time
ls.conn.eca_change([g_2000, g_2010, g_2020], "PC", distance=10_000, landscape_area=LA)

# Protected Connected land (Makurhini's MK_ProtConn, MK_ProtConnMult)
ls.conn.protconn(protected_areas, region, distance=[10_000, 50_000], transboundary=50_000,
                 bound=True, area_unit="ha", delta=True)
ls.conn.protconn_regions(protected_areas, ecoregions, id="eco_id", distance=10_000)
ls.conn.protconn_raster("protected_ids.tif", region, distance=10_000, transboundary=20_000)

# settlement probabilities of a spatial absorbing Markov chain (Makurhini's MK_dPC_SAMC)
g = ls.HabitatGraph.from_samc(patches_gdf, "resistance.tif", dispersal_distance=5_000, area_unit="ha")
g.node_importance("PC", pij_min=0.001)

# focal connectivity (Makurhini's MK_Focal_nodes): every node in its neighbourhood
ls.conn.focal_nodes(patches_gdf, id="patch_id", metric="PC", distance=5_000, search_buffer=10_000)

# grids (Makurhini's make_grid, MK_Connect_grid, MK_dECA_grid)
cells = ls.conn.make_grid(region, cell_area=1e8)            # hexagons of 100 km2
ls.conn.connectivity_grid(protected_areas, region, grid=cells, distance=10_000)
ls.conn.eca_change_grid([forest_2000, forest_2020], region, grid=cells, distance=5_000, metric="PC")

# nodes that could be added (restoration: 1 existing, 0 to add)
g = ls.HabitatGraph(areas, distances, restoration=flags)
g.node_importance(["PC", "IIC"], distance=10_000)      # + dPCres, dIICres of the nodes to add

g = ls.HabitatGraph.from_conefor("nodes.txt", "distances.txt", convention="conefor")
g.to_conefor("nodes.txt", "distances.txt")

# nodes from a raster of patch ids (or the patches of habitat classes), or from polygons
g = ls.HabitatGraph.from_raster("patches.tif", area_unit="ha", distance="edge", max_distance=20_000)
g = ls.HabitatGraph.from_raster("landcover.tif", habitat=[1, 2], directions=8)
g = ls.HabitatGraph.from_vector(patches_gdf, id="patch_id", area_unit="km2", distance="centroid")
imp = g.node_importance("PC", distance=10_000)
g.to_raster(imp, "dPC")            # node values on the patches (from_raster)
g.to_geodataframe(imp)             # node values with the geometries (from_vector)

g = ls.HabitatGraph.from_raster("patches.tif", distance="least-cost", resistance="resistance.tif")

ls.conn.node_table("patches.tif", area_unit="ha")              # Makurhini's nodesfile
ls.conn.distance_table(patches_gdf, id="patch_id", distance="edge", max_distance=5_000)  # distancefile
```

### Nodes and distances from maps

`node_table` and `distance_table` are Makurhini's `nodesfile` and `distancefile`
(Euclidean distances; the map units must be metres):

* **Raster of patch ids** (NA or `nodata` = no habitat; with `habitat=` the
  `directions`-connected patches of those classes are labelled first): the
  attribute is the number of cells × the cell width² in `area_unit` (or given per
  patch, `weighted` multiplies it by the area). Centroids are the means of the cell
  centres (`raster::xyFromCell`, averaged in 80-bit precision as R's `colMeans`);
  edge distances are the gap between the closest cells of two patches (what GEOS
  measures between the polygons of `terra::as.polygons`), 0 when they touch.
* **Polygons or points** (GeoDataFrame): the attribute is the area as sf's
  `st_area` computes it (GDAL's shoelace sums), or a column. Centroids are those
  of the largest part, or a point on the surface when that centroid falls outside
  (Makurhini's `st_centroid_within_poly`); edge distances come from GEOS.

* **Least-cost distances** (`distance="least-cost"`, `resistance=` a raster; NA cells
  cannot be crossed) are gdistance's `costDistance` as Makurhini calls it: between
  neighbouring cells (`cost_directions` 8 or 4) the conductance is 1 / the resistance of
  the cell of larger index (`cost_function="makurhini"`, its default `1/x[2]`) or
  1 / the mean of both (`"mean"`), divided by the distance between the cell centres;
  a step costs its reciprocal, and the distance between two nodes is the cheapest
  path between the cells of their centroids, in cost units (resistance × map units).
  Pairs without any path (other components of the resistance, nodes outside it or on
  NA cells) get their centroid distance, as in Makurhini (`unreachable="drop"` leaves
  them unconnected). Dijkstra runs from every node in parallel over the raster (each
  pair summed from its later node, as gdistance keeps the lower triangle), stops when
  the nodes before it are reached or at `max_distance`, and uses the same step costs
  (raster centre coordinates, R's x87 `sqrt`, `r_mean`): the distances are gdistance's
  to the last bit.
* **Commute-time distances** (`distance="commute-time"`, the same `resistance`,
  `cost_directions` and `cost_function`) are gdistance's `commuteDistance`: the expected
  time of a random walk over the same conductances from one node's cell to the other's
  and back, i.e. their effective resistance (as in Circuitscape) times the total
  conductance. The Laplacian of every connected part of the conductance graph, grounded
  at one node cell, is factorised once (SciPy's SuperLU) and solved for the other node
  cells in blocks; the distances agree with gdistance (CHOLMOD) to 1e-11. As in
  gdistance, nodes on NA cells or outside the resistance are left out; pairs in
  separate parts (where gdistance fails) get their centroid distance, or none with
  `unreachable="drop"`.

`max_distance` keeps the pairs up to that distance (in `distance_unit`), which is
what Makurhini's `threshold` does in pair tables (in its matrices it drops the
pairs up to it instead). Raster edge distances search the boundary cells of each
patch row by row, from the nearest row outwards, and stop at the gap of the
bounding boxes; with `max_distance`, only patches whose boxes come that close are
compared.

### Indices

| index | meaning | node importance |
|---|---|---|
| `NC`, `NL` | number of components and of links | `dNC`, `dNL` |
| `H` | Harary index: sum of 1 / (links between two nodes) | `dH` |
| `CCP`, `LCP` (`LCPnum`) | sum of the squared component attributes over the total attribute² / LA² | `dCCP`, `dLCP` |
| `IIC` (`IICnum`, `EC(IIC)`) | integral index of connectivity: ΣΣ a_i a_j / (1 + nl_ij) / LA² | `dIIC`, `dIICintra`, `dIICflux`, `dIICconnector` |
| `F`, `AWF` | flux and area-weighted flux: ΣΣ p_ij, ΣΣ a_i a_j p_ij | `dF`, `dAWF` |
| `PC` (`PCnum`, `EC(PC)`) | probability of connectivity: ΣΣ a_i a_j p*_ij / LA² (p* = most probable path) | `dPC`, `dPCintra`, `dPCflux`, `dPCconnector`, `varPC` |
| `A` | total attribute | `dA` |

`overall` also gives Makurhini's split of IIC and PC into intra-patch, direct and
stepping-stone parts (`PCintra(%)`, `PCdirect(%)`, `PCstep(%)`). A node's importance is
the loss of the index without it, in % (`varPC` is the loss of PC itself); dI =
intra + flux + connector. Without `landscape_area`, `IIC`, `PC` and `LCP` are NaN (only
their numerators, as in Makurhini) under the default convention, and use LA = 1 (as
Conefor) with `convention="conefor"`.

### Centrality

* `node_importance(["BC", "BCIIC", "BCPC"], ...)` (asked for by name, not in the
  default list) gives Conefor's betweenness centralities (Bodin & Saura 2010): `BC`,
  Freeman's betweenness of the links normalised by (n − 1)(n − 2) / 2; `varBC_IIC`, over
  ordered pairs the share of their shortest paths through the node times
  a_i a_j / (1 + nl_ij), and `dBC_IIC` in % of IICnum; `varBC_PC`, a_i a_j p*_ij of the
  pairs whose most probable path runs through the node, and `dBC_PC` in % of PCnum.
  They come from the same searches as the other indices (Brandes' accumulation).
* `centrality(distance, probability=None, binary=True)` is Makurhini's
  `MK_RMCentrality` (igraph): binary, the graph of the links, with `degree`; weighted
  (`binary=False`, or a `probability`), every pair with its probability p, 1 / p as
  the length of the edge and the weight of `strength` and eigenvector centrality.
  `eigen` is the leading eigenvector (the largest 1), `close` igraph's closeness
  (1 / the sum of the distances to the reached nodes; weighted, normalised: times
  their number), `BWC` igraph's betweenness (paths equal within 1e-10 all count),
  `cluster` the component. With python-igraph installed, `memb_rw` holds the walktrap
  communities (weighted by p, as in Makurhini); Makurhini's Louvain communities
  depend on a random order and are left out.

### Focal nodes

`ls.conn.focal_nodes(polygons, distance=, search_buffer=, metric="IIC")` is Makurhini's
`MK_Focal_nodes` (Latorre-Cárdenas et al. 2023). Around every node (the focal node),
the polygons intersecting its buffer of `search_buffer` (sf's 30 segments per quarter
circle; one buffer per distance, or one for all) form a graph: distances between their
points on surface, the landscape area the area of their bounding box. With three or more
nodes it gives the neighbourhood's `EC(IIC)` and `IIC` (or PC), the focal node's `dIIC`
and its intra/flux/connector fractions (Makurhini's `MK_dPCIIC` conventions) and the
composite connectivity index `IComp` = IIC × dIIC; with fewer, NaN. The attribute is the
area in `area_unit` (Makurhini: always ha), a column or values. Makurhini simplifies the
focal polygon with mapshaper before buffering it, which only matters for nodes at the
very edge of the buffer; landschaft buffers the polygon itself. Makurhini's
`fragmentation` statistics are not included.

### Spatial absorbing Markov chains (SAMC)

`HabitatGraph.from_samc(polygons, resistance, dispersal_distance=)` is Makurhini's
`MK_dPC_SAMC` (Fletcher et al. 2023, the samc package): the patches are connected by
the probability of settling in one when starting from the other, under a random walk
over the resistance raster. The cells with a resistance are the transient states; from
a cell the walk steps to a neighbour (`directions` 4 or 8, samc's order) with probability
(1 − absorption) · w / Σw, w = 1 / the mean resistance of both cells (samc's default;
`cost_function="makurhini"`: 1 / the neighbour's) over √2 for diagonal steps, and settles
with the cell's absorption: by default 0 in the patches and exp(−1.247 − 1.788 ·
log(dispersal distance / cell size)) elsewhere, or an `absorption` raster. With
F = (I − Q)⁻¹ (one sparse LU factorisation, solved for the vertex cells), settling in cell
d from cell o has probability F[o, d] · absorption[d]; between two patches it is the
`settle` (`"sum"` or `"mean"`) of those values from the boundary vertices of the later
patch to those of the earlier one (`boundary_vertices`: Makurhini's `SR_vertices`,
`vertices` 10 by default or a `vertex_fraction`, picked systematically). The graph has
`kind="probability"`; `node_importance("PC", pij_min=0.001)` gives `MK_dPC_SAMC`'s
values. `ls.conn.settlement_probabilities` returns the matrix itself. Makurhini's default
absorption rasterizes its own example data (`habitat_nodes`) instead of the given
patches; landschaft uses the given patches. Random vertex sampling and mapshaper's
simplification (`keep`) are not ported.

### Links, changes and nodes to add

* `link_importance(metric)` is Makurhini's `MK_dPCIIC_links` with `removal`: the loss
  of IIC or PC (in %) without each link (the links of the IIC graph, the edges above
  `pij_min` of the PC graph). Other pairs change nothing; Makurhini lists them too, with
  0 (PC) or, for IIC, the same share of its unconnected pairs as the links carry.
* `link_change(changes, metric)` is its `change`: for each pair whose value differs
  in `changes` (a pair table or a matrix), `100 (I - I') / I` (negative for a gain).
  For IIC a pair that comes closer than the distance gains a link and anything else
  counts 0; for PC the pair's edge takes the probability of its new value, without the
  `pij_min` cut, as in Makurhini.
* `restoration=` (1 existing, 0 to add, as Makurhini's column) rates the nodes to add:
  with the default convention `node_importance` adds `dPCres` / `dIICres`, the gain of
  each node on its own over the existing ones in % of the graph with it (Makurhini's
  code misaligns the attributes unless the nodes to add come last; landschaft uses
  the definition). With `convention="conefor"` it is Conefor's "adding nodes" mode:
  `overall` and the removal importances are those of the existing nodes, and every
  index gets `d...add` = its gain in % of the existing graph (`dAadd` ... `dPCadd`).

Removing a link repairs, in every source's shortest-path tree, the subtree below the
link (as for a node). Changes that shorten a pair and nodes to add use the lengths
between all nodes: the new length of a pair is `min(D_ij, D_ia + w + D_bj, ...)`, or
through the node to add. A longer pair is removed and added back with its new length.

### ProtConn

`ls.conn.protconn(protected_areas, region, distance=...)` is Makurhini's `MK_ProtConn`
(Saura et al. 2017, 2018), one row per distance: `ECA`, `PC`, `LA` (the region's area),
`Protected_surface`, `Prot`, `ProtConn`, `ProtUnconn`, `RelConn`, `Unprotected`, the
fractions `ProtConn_Prot`, `_Trans`, `_Unprot`, `_Within`, `_Contig` and their `_land`
shares, and with `bound=True` `ProtUnconn_Design` and `ProtConn_Bound`. The nodes
(`protconn_nodes`) are Makurhini's: the protected areas that touch the region, cut to
it, dissolved and split into separate polygons (their area in the region); with
`transboundary` the parts of them outside the region and the protected areas within
that distance (`transboundary_type="nodes"`: of the region's areas; `"region"`: of the
region), cut to that zone, are transboundary nodes (attribute 0); the region itself
joins them with `bound` (only next to transboundary nodes, as in Makurhini). Distances
run between polygon edges (GEOS) or are least-cost over a `resistance`, probabilities are
`exp(d log(p) / t)` without a cut, and `protconn_indicators` applies
`get_protconn_grid`'s formulas to any prepared nodes and distances. Makurhini weighs the
pairs with a transboundary node 1 instead of a_i a_j (`transboundary_weight=1`, the
default; 0 makes them stepping stones only). `delta=True` also rates each protected area
(`dProt`, `dProtConn`, `varProtConn` once removed; Makurhini weighs the transboundary
pairs 0 in the region's ECA but 1 without the area, landschaft uses one weight).
`protconn_regions` runs every region of a GeoDataFrame (`MK_ProtConnMult`). A region
with one dissolved node gets MK_ProtConn's fixed values for it.

Makurhini simplifies the region and the zones' polygons with mapshaper before
buffering them (Visvalingam, 5 % of the vertices); landschaft buffers the exact
polygons (30 segments per quarter circle, as sf), so both select the same areas when
the simplification keeps the shapes (as for the reference polygons).

### ProtConn from rasters and on grids

* `protconn_raster(protected_ids, region, ...)` is `MK_ProtConn_raster`: every
  protected-area id inside the region (rasterized on the protected areas' grid, cells
  whose centre is inside) is a node with the area of its cells there; transboundary
  nodes are the ids outside the region within `transboundary` (Makurhini keeps them only
  when there are two or more); distances run between the nodes' centroids. As in
  Makurhini, ProtConn[Within] sums the attributes of every node. (Makurhini pairs the
  attributes, sorted by id, with the centroids in the order the ids appear in the
  raster, and fails without two transboundary nodes; landschaft keeps each id with its
  centroid.)
* `make_grid(region, cell_area)` gives Makurhini's grid: sf's `st_make_grid(square =
  FALSE)` hexagons (the same coordinates) or raster squares over the region's extent,
  the cells touching the region (`clip`, `grid_boundary` as in Makurhini).
* `connectivity_grid(protected_areas, region, grid=|cell_area=, distance=)` is
  `MK_Connect_grid`: ProtConn of every cell (the cell as the region, the areas selected
  once for the whole region, the other cells' areas transboundary), or with
  `protconn_index=False` PC, ECA and normalised ECA of the patches dissolved in each
  cell. Makurhini's default `pij_min` becomes infinity there and cuts every link (each
  node alone): `pij_min=np.inf` reproduces it, the default keeps every link.
* `eca_change_grid(times, region, ...)` is `MK_dECA_grid`: the ECA change between
  consecutive times in each cell, with Makurhini's rules for cells where the habitat
  appears or disappears.

### ECA over time

`ls.conn.eca_change(graphs, metric, distance=, landscape_area=)` is Makurhini's `MK_dECA`:
for one graph per time (or scenario) the habitat area (the node areas of graphs made
by `from_raster` / `from_vector`, summed as R does, else the sum of the attributes),
the ECA (EC(PC) or EC(IIC), capped at LA), both in % of LA and of the habitat area,
`dA` and `dECA` (change from the previous time, the first from LA), `rECA = dECA / dA`
and Makurhini's reading of them ("+ Connectivity loss", "+ Habitat loss", ...). A graph
may hold a single node (its ECA is its attribute). Makurhini rounds some columns to
3 decimals (which ones depends on the number of times); landschaft does not round.

### Connections and conventions

`connections` holds distances (default), `kind="probability"` direct dispersal
probabilities, or `kind="link"` 0/1 links, as Conefor's connection types. A square
matrix is read from its upper triangle (NaN = not connected); in a pair table,
missing pairs are not connected. The probability of a distance d is the negative
exponential kernel `exp(d log(probability) / distance)`.

| | `convention="makurhini"` (default) | `convention="conefor"` |
|---|---|---|
| links (IIC, H, NC...) | pairs closer than `distance` (d < t) | pairs up to `distance` (d ≤ t) |
| `pij_min` | 0.01: probabilities below it are dropped | 0: Conefor's `minProb` heuristic (links at or below it dropped, paths no more probable not extended) |
| unconnected pairs in IIC | count as 16443701 links apart (MK_dPCIIC's value for them) in the whole graph, not in the graphs without a node | add nothing |
| without `landscape_area` | indices NaN, numerators reported | LA = 1 |
| `PC > 1`, `EC > LA` | capped, as MK_dPCIIC | not capped |

Makurhini's default distance (the median distance, `distance=None`) is also the
default here. Makurhini rounds its node importances to 7 decimals and its overall
fractions to 5 (sharing the rounding error out so they add to 100); landschaft
returns the unrounded values.

### How it works

Each shortest path is found once per source node, in parallel over the sources:
breadth-first search for links, Dijkstra over −log p for probabilities. The node
importances (dI) need the index of the graph without each node. Conefor and
Makurhini recompute the whole graph once per node; here, removing a node k only
changes the paths that ran through it, which are the subtree of k in each source's
shortest-path tree. Their new lengths come from a Dijkstra restricted to that
subtree, seeded from untouched neighbours. For links, most subtree nodes keep their
level through another neighbour and are skipped. For probabilities, the four best
ways into every node, kept while the tree is built, give most seeds without
scanning the neighbours. The per-node contributions are merged in source order, so
the results do not depend on the number of threads.

### Verification

* `tools/make_reference_conn.py` runs Makurhini's `MK_dPCIIC` (Makurhini 3.0) on its
  example `habitat_nodes` (404 patches, centroid distances) and three synthetic
  scenes (random points, non-metric costs, separate clusters) with IIC and PC,
  several distances, probabilities, `pij_min` and landscape areas
  (`tests/data/reference/conn/makurhini.csv.gz`). `tests/test_conn_reference.py`
  compares the unrounded values of its internal steps to 1e-12 and reproduces its
  rounded output exactly.
* The same script builds Conefor Sensinode 2.2's own index code (`landscape.cpp`,
  with `tools/conefor/conefor_harness.cpp` replaying its node-removal loop) with
  g++ and stores every index and node importance of the synthetic scenes
  (`conefor.csv.gz`). Conefor computes in single precision: the values agree to
  its precision (1e-5 relative for the indices).
* `make_reference_conn.py spatial` stores `nodesfile`, `distancefile` and `MK_dPCIIC`
  on Makurhini's example raster (`habitat_nodes_raster`, 398 patches) and on 53
  synthetic polygons (multipart, with holes, U-shaped, touching and overlapping) in
  `tests/data/reference/conn/spatial`. `tests/test_conn_spatial.py` finds the same
  bits for the node attributes and the centroid distances (rasters and polygons) and
  for the polygon edge distances; raster edge distances agree to the last bit
  (GEOS measures between terra's polygon vertices: 1 in 80,000 values differs by 1
  ulp). It also checks raster edge distances against brute force (non-square
  cells too) and GEOS on the cell polygons.
* Least-cost distances (`resistance_matrix`, 8 and 4 directions, `1/mean(x)`; the
  polygons over a resistance with NA walls, whose unreachable pairs take the centroid
  distance) are gdistance's bits, and `MK_dPCIIC` with them is reproduced exactly;
  `tests/test_conn_spatial.py` also checks them against SciPy's Dijkstra on the
  graph built from gdistance's formulas.
* `make_reference_conn.py commute` stores `distancefile(type = "commute-time")` on the
  polygons (8 and 4 directions, both cost functions) and on the example raster over
  `resistance_matrix`; `tests/test_conn_spatial.py` matches them to 1e-11 (the same
  pairs) and checks commute times against the Laplacian's pseudo-inverse.
* `make_reference_conn.py links` runs `MK_dPCIIC_links` (removal, and changes of 40
  pairs brought closer or moved apart) and `MK_dPCIIC` with `restoration` on the three
  synthetic scenes (`makurhini_links.csv.gz`): the same rounded values, restoration
  gains to 1e-12. The Conefor driver also replays its "adding nodes" mode (the last
  five nodes of each scene), matched to its precision for every index.
* `make_reference_conn.py deca` runs `MK_dECA` on three times of the polygons (PC and IIC,
  two distances) and of the example raster (`deca_steps.csv`, `makurhini_deca.csv.gz`):
  `dA` has the same bits and the other values agree to 1e-13.
* `make_reference_conn.py protconn` runs `MK_ProtConn` on 36 synthetic protected
  areas and two regions (without and with transboundary zones of both types, and
  ProtConn[Bound]) with its inner steps: fed Makurhini's own nodes and distances,
  `protconn_indicators` gives `get_protconn_grid`'s bits; landschaft's own nodes have
  the same areas (1e-14) and its rounded ProtConn table is Makurhini's, value for value.
* `make_reference_conn.py grids` stores `make_grid` (hexagons and squares: the same
  coordinates), `MK_Connect_grid` (ProtConn with its default and with `pij_min = 0`, and
  PC), `MK_ProtConn_raster` (three transboundary settings) and `MK_dECA_grid`; all are
  reproduced after Makurhini's rounding (the dECA grid to 1e-12).
* `make_reference_conn.py centrality` runs `MK_RMCentrality` (binary and weighted, with
  and without a probability) on the polygons and the example raster
  (`makurhini_centrality.csv.gz`): degree, BWC and cluster are the same, the other
  values agree to 1e-12 (eigenvectors: ARPACK against LAPACK).
* `make_reference_conn.py betweenness` runs the command line of Conefor 2.7.1
  (`CONEFOR_EXE`) for BC, BC(IIC) and BC(PC) on the synthetic scenes
  (`conefor_betweenness.csv.gz`), matched to the 6 digits it prints. Conefor keeps the
  probabilities in single precision: where a path is shorter than the direct link by
  less than float precision (near-collinear points, 1e-10 to 8e-8 relative) it counts
  the path; on the Euclidean scenes BC(PC) is 0 in exact arithmetic, and the tests
  check that every difference is such a tie.
* `make_reference_conn.py focal` runs `MK_Focal_nodes` (IIC and PC, one and two
  distances and search buffers, `pij_min`) on the polygons and the ProtConn areas
  (`makurhini_focal.csv.gz`): every neighbourhood is the same, the indices agree to
  1e-12 and the importances after Makurhini's 7-digit rounding. (Makurhini fails with a
  `threshold` there.)
* `make_reference_conn.py samc` runs `MK_dPC_SAMC` on the polygons over their resistance
  (with an absorption raster written from them), 4 and 8 directions, sum and mean,
  10 or 3 vertices or a fifth of them, and stores samc's settlement probabilities: they
  agree to 1e-14, and the PC importances are Makurhini's after its rounding.
* `tests/test_conn.py` checks every index and importance against a brute-force
  recomputation of the graph without each node (and with its attribute set to 0),
  under both conventions, and the input kinds, pair tables, Conefor files and
  thread independence.

### Performance

| graph | Makurhini 3.0 | landschaft |
|---|---|---|
| `habitat_nodes` (404 patches), 10 km, IIC / PC with node importance | 1.4 s / 10.6 s | 0.005 s / 0.017 s |
| 2,000 random patches, 9 km (47 k links; PC: 1.2 M pairs above `pij_min`), IIC / PC | – | 0.09 s / 1.7 s |
| 5,000 random patches, 6 km (134 k links; PC: 4.3 M pairs), IIC / PC | – | 0.8 s / 19 s |
| 4000×4000 raster, 8,395 patches: centroid distances, all 35 M pairs / up to 5 km | – | 0.9 s / 0.1 s |
| same, edge distances up to 5 km / 20 km / all pairs | – | 0.24 s / 1.1 s / 12.5 s |
| `habitat_nodes_raster` over `resistance_matrix` (222 k cells): least-cost distances | 10.2 s | 0.5 s |
| same: commute-time distances | 50 s | 3.7 s |
| 3000×3000 resistance, 300 nodes: least-cost distances, all pairs / up to 500 k cost units | – | 46 s / 1.5 s |
| 2,000 random patches: importance of every link, IIC (47 k links) / PC (1.2 M edges) | – | 0.1 s / 1.9 s |
| 5,000 random patches: importance of every link, IIC (134 k) / PC (4.3 M) | – | 0.9 s / 22 s |
| `habitat_nodes_raster` (398 patches), 20 km: centrality, binary / weighted (79 k edges) | 0.53 s / 1.1 s | 0.07 s / 0.14 s |

---

## 15. Circuit Connectivity (Circuitscape / Omniscape)

`ls.circuit` treats a conductance raster as an electrical circuit, as
[Circuitscape](https://circuitscape.org) 6.1 does, and `ls.omniscape` is
[Omniscape](https://github.com/Circuitscape/Omniscape.jl) 0.7 (omnidirectional
moving-window connectivity) on top of it.

```python
import numpy as np
import landschaft as ls

conductance = 1 / resistance            # NaN: no node
source = np.zeros_like(conductance)     # amps injected per cell
ground = np.zeros_like(conductance)     # conductance to ground; np.inf: direct ground (0 V)
source[10, 12] = 1.0
ground[80, 95] = np.inf

r = ls.circuit.advanced(conductance, source, ground)   # four_neighbors=False, n_threads=0
r.current   # current through every cell (0 where nothing flows)
r.voltage   # node voltages (NaN without a node)
```

### Advanced mode

It is Circuitscape's advanced mode on in-memory rasters (`compute_omniscape_current`,
the solve behind every Omniscape window):

* every cell with a conductance > 0 is a node, joined to its 8 neighbours (4 with
  `four_neighbors=True`) by the average conductance of both cells, `(x + y) / 2`, or
  `(x + y) / (2√2)` on the diagonals;
* a cell that is both a source and a ground loses its source; every connected component
  with a source and a ground is solved (finite grounds on the diagonal, direct grounds at
  0 V), the others carry no current;
* the grounded Laplacian is factorised exactly: a multifrontal Cholesky over a
  nested dissection of the grid (every separator line a dense front, factorised with
  Eigen's dense kernels); independent components in parallel;
* a node's current is the larger of its inflow and outflow, leaving out branch currents
  under 1e-8 of the largest one, with Circuitscape's column-major node order.

### Verification

`python tools/make_reference_omniscape.py advanced` (Julia with Circuitscape 6.1, see the
script) runs `compute_omniscape_current` with the `cholmod` and the default `cg+amg`
solvers on synthetic rasters: NoData islands, finite and direct grounds, components
without a source or a ground, 4 and 8 neighbours, an Omniscape window and conductances
over six orders of magnitude (`tests/data/reference/circuit`). The same cells carry
current and the values agree with CHOLMOD to 1.2e-13 of the largest current; the
iterative `cg+amg` itself is 2e-7 away from both.

### Omniscape

```python
ds = ls.omniscape(
    "resistance.tif", radius=100,     # window radius in cells
    block_size=5, buffer=0,           # Omniscape's options, same names
    source=None,                      # default: 1 / resistance (source_from_resistance, r_cutoff)
    flow_potential=True, normalized_current=True,
)
ds.cum_currmap                 # cumulative current
ds.flow_potential              # the same circuits with conductance 1
ds.normalized_cum_currmap      # cum_currmap / flow_potential
ds.cum_currmap.rio.to_raster("cum_currmap.tif")
```

Every `block_size` × `block_size` block whose sources add up to more than 0 is a target.
Its window is the circle of `radius + buffer` cells around it; the block's strength is
injected into the sources within `radius` (the block left out, shared in proportion to
their strengths) and flows to the grounded target cell. The window currents are summed
into `cum_currmap`. Omniscape's options map one to one: `source_threshold`, `r_cutoff`,
`resistance_is_conductance`, `four_neighbors` (`connect_four_neighbors_only`),
`correct_artifacts`, `mask_nodata`, `flow_potential` (`calc_flow_potential`) and
`normalized_current` (`calc_normalized_current`). Omniscape's quirks are kept on
purpose: windows and circles cut at the map edges, the target block zeroed at
`radius + buffer` cells from the window's corner (not at the target when the window is
cut on the left or the top), even block sizes made odd, and the block-artifact
correction.

How it is fast:

* each window is factorised by a multifrontal Cholesky built on a geometric nested
  dissection of the grid: the window is cut by its middle row or column, the halves
  recursively, and every separator line becomes a dense front whose pivots are
  factorised with dense kernels (LLᵀ, triangular solve, rank-k update) and whose Schur
  complement goes to the parent front. On a disk of radius 100 (31 k cells) that is
  12 ms against 26 ms with Eigen's sparse LDLᵀ in the same order (and 67 ms / 119 ms
  for Circuitscape's CHOLMOD / cg+amg);
* windows of the same shape (the interior disk, and each shape cut by the edges) share
  one symbolic structure between all threads; a window only refills its values; the
  forward substitution runs inside the factorisation, while each front is in cache;
* the flow potential has conductance 1 everywhere: its factorisation is done once per
  shape and thread, and the flow potentials of up to 8 targets are solved together;
* on x86-64 the circuit code is also built for AVX2 and FMA (`landschaft._circuit_avx2`,
  loaded only when the CPU has them; `LANDSCHAFT_CIRCUIT_AVX2=0` keeps the baseline
  build): the dense kernels are 1.2-1.5x faster. Both builds give the same results to
  1e-11 and are each deterministic (buffers aligned to 64 bytes);
* windows are solved in parallel and added to the maps row by row in a fixed order, so
  the result does not depend on the number of threads;
* very large windows: each thread keeps the factor of a window (about 145 MB at radius
  300, plus twice that for the flow potential), so fewer threads run when they would not
  fit in `memory_limit` (default: half of the physical memory; `attrs["threads"]` tells
  how many ran).

Verification: `python tools/make_reference_omniscape.py omniscape` runs Omniscape 0.7's
`run_omniscape` (INI files, `cholmod` and `cg+amg`) on its own test rasters with its
test configurations 1-3, and on synthetic maps (NoData blobs and lines, source files with
NoData, block sizes 1-7, buffers, 4 neighbours, conductance input, windows wider than
the map). NoData and zero cells are the same, and every value agrees with CHOLMOD to
1.7e-11 (relative). Omniscape's default `cg+amg` is up to 1.6e-4 of the largest value away
from its own CHOLMOD result where the block correction amplifies its error.

### Conditional connectivity, reclassification, INI files and single precision

```python
# Sources connect only to targets of similar climate (Omniscape's conditional options):
ds = ls.omniscape(res, 50, block_size=5,
                  condition1=temperature_now, condition1_future=temperature_2050,
                  compare_to_future="1", comparison1="within",
                  condition1_lower=-1.0, condition1_upper=1.0,
                  condition2=landcover, comparison2="equal")

ds = ls.omniscape(res, 50, reclass_table={1: 10, 2: 1, 3: None})   # None / NaN: NoData
ds = ls.omniscape(res, 50, precision="single")                      # float32 in and out

ds = ls.circuit.omniscape_ini("config.ini")    # an Omniscape INI file, outputs in project_name
```

* Conditions (`conditional`, `n_conditions`, `compare_to_future`): after a window's
  sources are scaled, those whose present condition does not match the target block are
  set to 0 (so less than the block's strength may be injected, as in Omniscape).
  `"within"` keeps a source when `lower <= condition - median(block) <= upper`, `"equal"`
  when `condition == mode(block)` (StatsBase's mode: the first value, in column-major
  order, to reach the highest count); the block's value comes from the future condition
  when `compare_to_future` names it, else from the present one; NoData conditions keep the
  source.
* `reclass_table` (`reclassify_resistance`): rows (value, new value), each matched against
  the original values; the default sources come from the reclassified resistance.
* `precision="single"` reads the rasters as float32 and returns float32 maps, as
  Omniscape does; the arithmetic stays double (Omniscape's own float32 solves are up to
  2e-4 away from the double result).
* `ls.circuit.omniscape_ini(path)` reads every Omniscape option with its default
  (`parse_ini` is Omniscape's parser), the rasters (paths from the working directory, or
  next to the INI file) and the reclass table, checks sizes and projections, and writes
  `config.ini`, `cum_currmap`, `flow_potential`, `normalized_cum_currmap` and
  `classified_resistance` to `project_name` (`_1`, `_2`... when it exists), GeoTIFF or
  ASCII grid with NoData -9999. `solver`, `parallelize` and `parallel_batch_size` have no
  effect.

Verification: `python tools/make_reference_omniscape.py ini` runs Omniscape's own test
INI files (configurations 1-7, 4a-4c, 32 bit and reclass, with its test rasters, copied
to `tests/data/reference/circuit/omniscape_input`) through `run_omniscape`: every output
agrees with Omniscape's CHOLMOD run to 5e-13 (conditional `within` and `equal`, futures,
four neighbours, conductance input, reclassification), and the single-precision run to
Omniscape's float32 rounding.

Timings on an Intel i5-13600K (6 performance + 8 efficiency cores, 20 threads), random
log-normal resistance, source = 1 / resistance, flow potential and normalised current
(`tools/omniscape_window_benchmark.jl` times single windows):

| map | Omniscape 0.7, 20 threads (cg+amg / cholmod) | landschaft, 1 thread / 20 threads |
|---|---|---|
| 300×300, radius 25, block 3 (10,000 targets) | 17.4 s / 18.6 s | 7.2 s / 0.76 s |
| 800×800, radius 10, block 3 (70,756 targets) | 29.4 s / – | 6.5 s / 0.64 s |
| 700×700, radius 25, block 5 (19,600 targets) | 46.7 s / – | 14.3 s / 1.36 s |
| 1000×1000, radius 100, block 15 (4,356 targets) | 275 s / – | – / 12.4 s |

The results agree with Omniscape's `cg+amg` to its tolerance (1e-6 of the largest
current). Large windows are limited by memory bandwidth with many threads (a radius-100
window writes and reads a 13 MB factor): 6x from 1 to 20 threads there, 10x for radii
10-25.

### Circuitscape (raster modes)

```python
from landschaft.circuit import circuitscape_advanced, circuitscape_ini

r = circuitscape_advanced("resistance.tif", "sources.asc", "grounds.asc",   # paths, DataArrays or arrays
                          polygons="regions.asc", four_neighbors=False, avg_resistances=False,
                          ground_is_resistance=True, remove_src_or_gnd="keepall")
r.voltage, r.current, r.nodemap

# effective resistances between focal points (a raster of ids; several cells of one id: a region)
p = circuitscape_pairwise("resistance.tif", "sites.asc", included_pairs="pairs.txt", cum_map=True)
p.resistances          # ids in the first row and column, -1 where not connected
p.cum_current          # the pairs' current maps summed

# one-to-all (all_to_one=True: all-to-one): every focal point in turn against the others
o = circuitscape_one_to_all("resistance.tif", "sites.asc", strengths="strengths.txt", cum_map=True)
o.resistances          # ids and the voltage over the injected strength at each point

r = circuitscape_ini("job.ini")      # a Circuitscape INI file; maps written next to output_file
```

Circuitscape 6.1 on rasters, with its conventions: the node map (habitat cells numbered in
column-major order; the cells of a short-circuit region that holds habitat merged into one
node, its non-habitat cells included), edges weighted by the average conductance or, with
`connect_using_avg_resistances`, the average resistance of the two cells (summed between
merged nodes), connected components that ignore edges of weight 0, the mask, reclass
tables applied to the values read, sources and grounds summed per node (maps or text
lists of `value x y`), grounds as resistances (0: direct ground), unit currents, direct
grounds and the four `remove_src_or_gnd` policies with their quirks. ASCII grids are read
in double, as Circuitscape does (GDAL would give float32). Each component is factorised by
the same multifrontal Cholesky as Omniscape's windows, with the short-circuit regions
eliminated last. Large components (an L of more than 32 MB) are factorised in parallel:
subtrees of the dissection of similar work on separate threads, each with its own stack
of update matrices; the fronts above them by heights (several at once, or a lone one with
parallel kernels); large fronts (128 pivots or more) by a blocked right-looking Cholesky
over fixed blocks (LLT of 128-column diagonal blocks, panel solves by 256-row chunks,
the trailing update by 256 × 128 tiles). Every front is built in an aligned workspace,
so the arithmetic, and the result, is the same bits with any number of threads; a
forward substitution fused into the factorisation keeps the single-thread order of its
operations (the contributions of a subtree to the rows outside it are replayed at its
root's turn). On a 1000×1000 raster the factorisation takes 0.7 s with 20 threads
instead of 3.4 s (advanced mode as a whole: 2.6 s instead of 5.2 s).

Implemented: `scenario = advanced` (voltage and current maps; `circuitscape_ini` writes
`<output>_voltmap` and `<output>_curmap`, ASCII grids or GeoTIFF, and the configuration
in `output_file`) and `scenario = pairwise`: focal points from a raster or a text list
(sorted by id, negative ids dropped), include/exclude pair files (a `min`/`max` matrix or
a `mode` list; include mode drops the points it does not name), resistances with
Circuitscape's rules (-1 between components, 0 between focal points on one node), every
pair's voltage and current maps, the cumulative and maximum current maps with
`set_focal_node_currents_to_zero`, `set_null_*_to_nodata` and `log_transform_maps`, and
`<output>_resistances.out` / `_resistances_3columns.out`. Each component is factorised
once, with one node grounded (the voltage differences are exact; Circuitscape solves the
singular Laplacian shifted by 10ε), and its pairs are solved in parallel batches of right-
hand sides; without maps and exclusions Circuitscape's shortcut applies (only the pairs of
the first focal point are solved, the others follow from its voltages, with its quirks).
Focal regions (several cells of one id) get a graph per pair, the two regions merged with
the polygons they touch, as Circuitscape does; the pairs run in parallel.
`scenario = one-to-all` / `all-to-one`: the focal points and regions are merged into the
short-circuit regions; for every focal point one-to-all injects 1 A (or its variable
source strength) there with every other point a direct ground, all-to-one grounds it
and injects at the others; only its component is solved (the points in parallel, on one
graph unless focal regions meet an include file); results `id, resistance` (one-to-all:
voltage over strength, -1 when not solved; all-to-one: 0), per-point voltage and current
maps (`_voltmap_<id>`, `_curmap_<id>`), cumulative and maximum maps. Circuitscape's
quirks are kept: the include file prunes the points in both of its modes, excluded
points still inject current in all-to-one, variable strengths are taken by position,
and a point alone on its map is skipped.

Verification: `python tools/make_reference_circuitscape.py` copies Circuitscape's raster
test inputs (`tests/data/reference/circuit/circuitscape/input/raster`, MIT) and runs every
one of its INI files with the `cholmod` solver; in advanced mode (`mgVerify1`-`7`,
`ex_advanced`: polygons, sources and grounds, finite and direct grounds, 4 and 8
neighbours, average resistances, a 355×481 map) every map agrees to 5e-12 of its largest
value, with the same zero cells. In pairwise mode (`sgVerify1`-`17`: polygons, focal
regions, masks, included pairs, 4 and 8 neighbours, voltage, current, cumulative and
maximum maps) the resistances agree to 1e-12 (to 9e-10 relative on `sgVerify17`,
resistances up to 3e3, where CHOLMOD's shifted singular solve itself loses digits) and
every map to 3e-13, with exactly the same set of per-pair maps. In one-to-all and
all-to-one (`oneToAllVerify1`-`13`, `allToOneVerify1`-`12`: polygons, focal regions,
include files, variable strengths, masks, focal-node currents set to zero) the results
and every map agree to 5e-14.

Timings (i5-13600K, 20 threads; a 1000×1000 log-normal resistance ASCII grid, 20 focal
points; reading and writing the ASCII grids included; Circuitscape 6.1 with its default
cg+amg, after compilation):

| run | Circuitscape | landschaft |
|---|---|---|
| pairwise, resistances only | 8.7 s | 2.7 s |
| pairwise with the cumulative current map (190 pairs) | 76.5 s | 9.5 s |
| one-to-all with every point's current map | 21.8 s | 4.6 s |

The results agree with Circuitscape's cg+amg to its tolerance (resistances to 7e-7). In
one-to-all and all-to-one the focal points are the only nodes with sources or grounds, so
every component is factorised once without them and each point is solved through the
dense Schur complement on the focal nodes (Circuitscape factorises the graph once per
point). ASCII grids are written by a parallel C++ formatter (shortest round-trip values).

### Circuitscape (network mode)

```python
from landschaft.circuit import circuitscape_network_pairwise, circuitscape_network_advanced

# edge list: node, node, resistance (graph_is_resistance=False: conductance)
r = circuitscape_network_pairwise("graph.txt", "focal_nodes.txt", current_maps=True)
r.resistances                        # ids in the first row and column, -1: not connected
r.maps[("voltages", 1, 5)]           # node, voltage of one pair (voltage_maps=True)
r.cum_node_currents, r.cum_branch_currents
a = circuitscape_network_advanced("graph.txt", "sources.txt", "grounds.txt")
a.voltages, a.node_currents, a.branch_currents
circuitscape_ini("network.ini")      # data_type = network, pairwise or advanced
```

Circuitscape 6.1's `data_type = network` with its conventions: nodes numbered from 1, or
from 0 (then every number, focal nodes, sources, grounds and pair files included, moves
up by 1, as Circuitscape.jl does; focal nodes from 0 move on their own too); duplicate
edges add their conductances (`sparse` in input order), then `A + Aᵀ` drops the edges
whose conductances add up to 0 and self loops are left out of the Laplacian; components
come from the remaining edges. Pairwise solves every pair of focal nodes per component
(no resistance shortcut: Circuitscape keeps it for rasters), resistances with the -1 / 0
rules of the raster mode, include / exclude pair files; with any map option every pair's
node currents (`max(inflow, outflow)`, Circuitscape's 1e-8 thresholds) and branch currents
(every edge, zeroed below 1e-8 of the largest, written above 1e-6) and, with
`voltage_maps`, its node voltages; node and branch currents summed over the pairs (the
first input line of an edge takes the sum). Advanced mode: sources and grounds by node
(`sources[node] = value`: the last line of a node wins), ground resistances or
conductances, direct grounds, the four conflict policies; node and branch currents of
the whole graph; `use_unit_currents` and `use_direct_grounds` do not apply to networks.
Files are written with Circuitscape's names (`_resistances.out`, `_voltages_<a>_<b>.txt`,
`_node_currents_cum.txt`, `_branch_currents.txt`...).

Graphs have no geometry, so the solver orders them by a nested dissection of their own:
breadth-first levels from a pseudo-peripheral node, the smallest level that leaves a
fifth of the nodes on either side as separator (trimmed of the nodes that touch only one
side), components apart; the sets of one depth are split in parallel and the tree is the
same with any number of threads. The fronts are those of the raster solver.

Verification: `python tools/make_reference_circuitscape.py network` runs Circuitscape's
network test cases (`sgNetworkVerify1-3`, `mgNetworkVerify1-3`) and six synthetic ones
(numbering from 0 and 1, duplicate edges and self loops, edges that cancel, focal nodes
repeated, isolated or out of the graph, include / exclude pairs, finite and direct
grounds, `rmvgnd` / `rmvall`, a 4000-node graph) with CHOLMOD: every output file, the same
set of files, to 1.1e-14. On a 1000×1000 grid-like network (1 M nodes, 3 M edges, 20
focal nodes, resistances only; 20 threads) landschaft takes 5.0 s, Circuitscape 27.7 s
with CHOLMOD and 182 s with its default cg+amg. The resistances agree with CHOLMOD's to
5e-10 (relative); an exact sparse LU of the grounded Laplacian (scipy, no diagonal shift)
differs from both by as much: that is the conditioning of a 1 M-node Laplacian and
Circuitscape's 10ε shift, not the solver.

---

## 16. Neutral Landscape Models (NLMR)

`ls.nlm` simulates neutral landscapes, the null models of landscape ecology, as
the R package NLMR does, and returns `xarray.DataArray`s on NLMR's grid (origin
0, 0; `resolution` cell size). With the same `seed` (NLMR's `user_seed`) each
model gives NLMR's landscape bit for bit: R's generator (`set.seed`, `runif`,
`rnorm` by inversion, `sample`) and the libstdc++ distributions of NLMR's C++
code (`std::mt19937`) are reproduced. `seed=None` draws a seed and stores it in
`attrs["seed"]`.

```python
import landschaft as ls

r = ls.nlm.random(200, 150, seed=42)                     # uniform values, rescaled to 0-1
p = ls.nlm.percolation(200, 150, prob=0.4, seed=1)        # 0/1 (uint8)
g = ls.nlm.planar_gradient(200, 150, direction=45)        # direction=None: drawn from the seed
e = ls.nlm.edge_gradient(200, 150, seed=3)
d = ls.nlm.distance_gradient(200, 150, origin=(10, 20, 30, 40))
classes = ls.nlm.classify(r, weighting=[0.5, 0.3, 0.2])   # int32 classes 1..3 (0 = missing)
jenks = ls.nlm.classify(r, n=4)                           # Jenks natural breaks
x = ls.nlm.rescale(some_array)                            # NLMR's util_rescale
m = ls.nlm.mpd(257, 257, roughness=0.5, seed=1)           # diamond-square (NLMR's grid size)
n = ls.nlm.neigh(200, 150, p_neigh=0.7, p_empty=0.1, categories=4, seed=2)
rc = ls.nlm.random_rectangular_cluster(200, 150, minl=5, maxl=20, seed=3)
cl = ls.nlm.random_cluster(200, 150, p=0.4, ai=[0.3, 0.3, 0.4], seed=4)
cu = ls.nlm.curds([0.5, 0.4, 0.6], [8, 8, 8], wheyes=[0.1, 0.1, 0.1], seed=5)  # 512 x 512
gf = ls.nlm.gaussian_field(200, 150, autocorr_range=20, mag_var=5, nug=0.2, seed=6)
fb = ls.nlm.fbm(200, 150, fract_dim=1.2, seed=7)
mf = ls.nlm.mosaic_field(200, 150, n=20, seed=8)          # dict with collect=/infinit=
```

NLMR behaviours kept with `nlmr_compat=True` (the default where they apply):

- `distance_gradient`: `origin` is applied as `(first row, last row, first column,
  last column)`, as NLMR indexes the raster; `terra::distance` treats rasters of at
  most 360 x 90 cells as longitude/latitude (terra's default CRS for such an extent)
  and returns haversine distances in metres; larger rasters get GDAL's proximity
  algorithm (cell units, float32). `nlmr_compat=False` gives exact Euclidean
  distances times `resolution`, with `origin = (first column, last column, first
  row, last row)`.
- `mpd`: NLMR's grid is a square of side `max(nrow, ncol) + 1` (made odd) minus one
  cell on each side, so the result is square (a warning says when it differs from
  `nrow x ncol`); `nlmr_compat=False` builds a `2**k + 1` grid and crops it to the
  size asked for. The normal deviates are drawn in NLMR's order (a fresh
  `std::normal_distribution` per cell), their log/sqrt and each step's cells are
  computed in parallel (in order on a torus, where cells of one step can meet).
- `neigh`: categories are filled from `categories - 1` down to 0, a neighbour
  "holds a category" when it is non-zero, the edges wrap, `p_neigh`/`p_empty` are
  rounded to float as in NLMR's C++.
- `random_rectangular_cluster`: a count of empty cells replaces NLMR's scan of the
  whole matrix after every rectangle (same result).
- `random_cluster`: clusters numbered as `terra::patches` does (in the order of
  their first cell), the empty cells filled from the clustered map (not from the
  cells filled before them), ties by `sample()`, cells without typed neighbours by
  `sample(prob = ai)`; neighbours are counted in parallel, the draws follow in order.
- `curds`: `dplyr::sample_frac` = `sample.int(n, round(n * share))` over the cells in
  row-major order; 1 is habitat (curds), 0 matrix.
- `gaussian_field`, `fbm`: NLMR 1.4 disabled `nlm_gaussianfield` and `nlm_fbm` (only a
  message) and ships RandomFields-free methods as the internal `simulate_gaussian_field`
  (circulant embedding) and `simulate_fbm` (1/f spectrum); these are reproduced and
  rescaled as the previous `nlm_*` did. That takes R's `fft` (Singleton's algorithm,
  ported statement by statement; the columns' and rows' transforms run in parallel
  with the same bits), R's `exp` and `x^y` on Windows (mingw-w64's x87 `exp` and
  `powl`), R's vector `sqrt` and the 80-bit `mean`/`sd`.
- `mosaic_field`: spatstat's `rpoislinetess(4)` (R's `rpois`, `runif`, x87 `sin`/`cos`)
  and `rMosaicField`: the tiles are the cells of the line arrangement in the unit
  square in `chop.tess` order (below before above each line, lines in order), each
  pixel centre (`as.mask`'s grid) takes its tile's `rnorm` value; `infinit` is
  `log(exp(Z))` of `rLGCP`'s circulant embedding (`grid.prep`, `covariance.prep`,
  `generate.normals` with R's `fft`). Kept NLMR quirks: `n = 1` sums three fields
  (`for (i in 2:n)`), `collect` divides the i-th field by `sqrt(i)` and rescales all
  steps with the first step's range.
- `classify`: `sum`/`cumsum` of the weights in 80-bit precision as R; when all
  class boundaries are equal every cell becomes class 1; Jenks' breaks cost
  O(classes x cells^2) as in NLMR (parallel over the rows of each class column).

Reference: `tools/make_reference_nlm.py` (NLMR 1.4.0 installed from a clone into a
private R library, R 4.4.2; the libstdc++ distributions checked against a probe
built with the Rtools g++), tests in `tests/test_nlm.py` and
`tests/test_nlm_rng.py`.
