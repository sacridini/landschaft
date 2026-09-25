; SPDX-License-Identifier: GPL-3.0-or-later
;
; GDL's WRITE_TIFF, GEOTIFF=... patches the GeoTIFF tags into the written file
; with UPDATEGEOTAGSINIMAGE, which overwrites a few image bytes (IDL writes
; them correctly). This replacement leaves the image untouched and stores the
; georeferencing in a world file (<name>.tfw) instead, which GDAL reads, so
; GTB's gdal_translate/gdalbuildvrt steps place the tiles as with IDL.
pro updategeotagsinimage, filename, geotiff
  compile_opt idl2
  tags = tag_names(geotiff)
  if total(tags eq 'MODELPIXELSCALETAG') eq 0 or total(tags eq 'MODELTIEPOINTTAG') eq 0 then return
  sc = double(geotiff.modelpixelscaletag) & tp = double(geotiff.modeltiepointtag)
  ;; tie point (I, J, K, X, Y, Z) at the upper-left corner (PixelIsArea); world files use pixel centres
  cx = tp[3] + (0.5d - tp[0]) * sc[0] & cy = tp[4] - (0.5d - tp[1]) * sc[1]
  dot = strpos(filename, '.', / reverse_search)
  tfw = (dot gt 0 ? strmid(filename, 0, dot) : filename) + '.tfw'
  openw, lun, tfw, / get_lun
  printf, lun, sc[0], 0d, 0d, -sc[1], cx, cy, format = '(G0.17)'
  free_lun, lun
end
