; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Accounting" for the landschaft test-suite.
; The size classes the interactive 'accounting' event proposes (scaled_cl_arr),
; copied verbatim from guidostoolbox.pro; written to outfile, one per line.
pro gtb_accounting_scaled_classes, input, outfile
  im = read_tiff(input) & sz = size(im,/dim)
  ext=bytarr(sz[0]+10,sz[1]+10)
  ext[5:sz[0]+4,5:sz[1]+4]=im eq 2b & tot_area = total(ext,/double)
  nw_ids = label_region(ext, all=1, / ulong) ;; use the default 8-conn here. Should the user select 4 then we need to redo nw_ids
  tot_ids = max(nw_ids) ;; total # of objects, the first one does not count because it is the background
  obj_area = histogram(nw_ids, reverse_indices = revind, / l64)
  obj_area = long64(obj_area)  ;; GDL: /l64 is ignored
  oba_max = max(obj_area[1:*]) ;; neglect background and get the area of the largest object
  obj_area[0] = 9000000000000000000  ;; set BG to a high number so we can search for size classes in one statement
  oba_min = min(obj_area) ;; area of the smallest object
  oba_range = oba_max - oba_min

  ;; scaled_cl_arr: thesholds based on the size range [1, max_object_area]
  scaled_cl_arr = strarr(6) & scaled_cl_arr[0] = 'X'
  scaled_cl_arr[1:5] = strtrim(ulong64(oba_min + [0.001, 0.01, 0.1, 0.5, 0.9]*oba_range),2)

  openw, u, outfile, /get_lun & printf, u, transpose(scaled_cl_arr[1:5]) & free_lun, u
end
