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
* **Accounting** (IDL-only in GTB): Tested against GTB's own IDL code (`tests/gtb_idl`, statements copied verbatim from `guidostoolbox.pro`) run in GDL. Maps and integer statistics are identical. Stored digests (`tests/data/reference/idl_reference.json`) cover 7 maps × 5 parameter sets, and random scenes run live when GDL is available.

To run the verification suite:
```bash
pytest -v
```
