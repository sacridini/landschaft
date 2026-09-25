# Reference outputs from the GuidosToolbox desktop application

Some GTB tools rely on IDL routines that GDL does not provide (`WATERSHED`,
`MORPH_DISTANCE`), so their reference outputs come from the GTB desktop
application (which bundles the IDL runtime). They are stored in
`tests/data/reference/gtb_desktop` by `tools/import_gtb_desktop.py`.

Inputs: `gtb_refs/input/input.tif` (GTB's sample map) and
`gtb_refs/input/influ_test.tif` (`scenes.blobs(42, shape=(300, 300), scale=22,
cover=0.3, missing=0.03)`). Save every output to `gtb_refs/out`. Paths must
not contain spaces (GTB refuses them).

1. **Euclidean Distance (batch)**: *File → Batch Process → Distance →
   Euclidean Distance*, select both files in `gtb_refs/input`, keep the
   defaults. GTB writes a `batch_...` folder next to `input`; copy its
   `*_dist.tif` files to `gtb_refs/out`.
2. **Influence Zones** on `input.tif` (*File → Read Image → GeoTIFF*, then
   *Image Analysis → Distance → Influence Zones*):
   - defaults (area 5000, CoreZone 0, Background 0): *File → Save Image →
     GeoTIFF* as `input_influ_5000_0_0.tif`;
   - CoreZone 2, Background 10: save as `input_influ_5000_2_10.tif`.
3. **Proximity** on `input.tif` (*Image Analysis → Distance → Proximity*),
   defaults: save as `input_prox_5000_0_0.tif` (GTB also writes
   `..._viewport.tif`) and copy `input_proximity_5000_0_0.csv` from
   `C:\GuidosToolbox\guidos_progs\mspatmp`.
4. The same on `influ_test.tif` with area threshold **1**:
   `influ_test_influ_1_0_0.tif`, and Proximity `influ_test_prox_1_0_0.tif`
   (+ viewport) with `influ_test_proximity_1_0_0.csv`.
