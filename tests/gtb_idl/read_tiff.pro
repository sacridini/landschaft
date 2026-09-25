; SPDX-License-Identifier: GPL-3.0-or-later
;
; GDL's TIFF_READ cannot read 8-bit paletted images ("Palettes only supported
; for 3 channel images with 16 bits per pixel"), such as the MSPA maps GTB
; reads back after running mspa. This replacement of READ_TIFF falls back to
; Python/rasterio (LANDSCHAFT_PYTHON, default 'python') for those files and
; returns the pixel values, as IDL does.
function read_tiff, filename, r, g, b, geotiff=geotiff, image_index=image_index, sub_rect=sub_rect
  compile_opt idl2
  catch, err
  if err ne 0 then begin
    catch, /cancel
    py = getenv('LANDSCHAFT_PYTHON') & if py eq '' then py = 'python'
    raw = filename + '.raw'
    code = 'import sys, rasterio; d = rasterio.open(sys.argv[1]); d.read(1).tofile(sys.argv[2]); print(d.width, d.height)'
    ;; cmd.exe strips the outer quotes of the whole command line
    spawn, '""' + py + '" -c "' + code + '" "' + filename + '" "' + raw + '""', out, err, /hide
    dims = long(strsplit(out[n_elements(out) - 1], ' ', /extract))
    im = bytarr(dims[0], dims[1]) & openr, lun, raw, / get_lun & readu, lun, im & free_lun, lun
    file_delete, raw
    return, im
  endif
  if n_params() gt 1 then $
    return, tiff_read(filename, red=r, green=g, blue=b, geotiff=geotiff, image_index=image_index, sub_rect=sub_rect)
  return, tiff_read(filename, geotiff=geotiff, image_index=image_index, sub_rect=sub_rect)
end
