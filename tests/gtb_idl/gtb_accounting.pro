; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Accounting" for the landschaft test-suite.
;
; The processing statements are copied verbatim from the 'batch_accounting'
; event of guidostoolbox.pro (https://github.com/ec-jrc/GTB). Only the GUI,
; logging, the IDL save/restore of obj_area (replaced by a copy) and the
; output format (raw arrays and a key=value text file instead of GeoTIFF and
; txt/csv reports) differ. GDL ignores HISTOGRAM's /L64 keyword, so each
; histogram is converted with long64() to get the 64-bit counts IDL returns
; (the 9e18 background sentinel below needs them). Run it with GDL or IDL:
;
;   gtb_accounting, 'in.tif', 'outdir/', conn8, cl1, cl2, cl3, cl4, cl5, big3pink
;
; Outputs in outdir: classes.raw (uint8), ids.raw (int32), pixels.raw (int32),
; all in the row order of the input file, and stats.txt.

pro gtb_accounting, input, outdir, conn8, cl1, cl2, cl3, cl4, cl5, big3pink
  cl1 = ulong64(cl1) & cl2 = ulong64(cl2) & cl3 = ulong64(cl3) & cl4 = ulong64(cl4) & cl5 = ulong64(cl5)

  im = read_tiff(input)  & sz = size(im,/dim)
  ;; rotate the image to be compliant with Guidos results!
  im = rotate(im,7)

  ;; extend the image with background so label_region can do a flood-fill in the BG
  ext = bytarr(sz[0]+10,sz[1]+10)
  ext[5:sz[0]+4,5:sz[1]+4]=im eq 2b & tot_area = total(ext)
  cl = byte([103, 33, 65, 1, 9, 17]) ;; colors to be used for the 6 size classes
  cl_name = ['black', 'red', 'yellow', 'orange', 'brown', 'green']
  nw_ids = label_region(ext, all=conn8, / ulong)
  tot_ids = max(nw_ids) ;; total # of objects, the first one does not count because it is the background
  obj_area = histogram(nw_ids, reverse_indices = revind, / l64)
  obj_area = long64(obj_area)  ;; GDL: /l64 is ignored
  oba_max = max(obj_area[1:*])
  obj_area[0] = 9000000000000000000  ;; set BG to a high number so we can search for size classes in one statement

  ;; ext2 = nw_ids in original size
  ext2 = long(nw_ids[5:sz[0]+4,5:sz[1]+4])

  ;; statistical summary of nw_id image
  cat = [cl1, cl2, cl3, cl4, cl5] ;; the defined size category thresholds
  cat_idlast = 0 & cat_arealast = 0
  q = where(cat GE 1, /l64)
  cat = cat[q] & cat = cat(uniq(cat))  ;; remove potential double entries
  nr_cat = n_elements(cat)
  cat_ID = cat*0    ;; # of objects in each cateory
  cat_area = cat*0   ;; total area of objects in each cateory
  oba = histogram(ext2, / l64) ;; oba has only the valid objects !!!
  oba = long64(oba)  ;; GDL: /l64 is ignored
  oba = oba[1:*] ;; remove the background

  ;; first category
  q = where(oba LE cat[0], ct, /l64)
  if ct gt 0 then begin
    cat_id[0] = ct & cat_area[0] = total(oba[q],/double)
  endif
  ;; other categories
  for idx = 1, nr_cat do begin
    if idx lt nr_cat then begin ;; not the last category
      q = where(oba GT cat[idx-1] and oba LE cat[idx], ct, /l64)
      if ct gt 0 then begin
        cat_id[idx] = ct & cat_area[idx] = total(oba[q],/double)
      endif
    endif else begin ;; the last category
      q = where(oba GT cat[idx-1], ct, /l64)
      if ct gt 0 then begin
        cat_idlast = ct & cat_arealast = ulong64(total(oba[q],/double))
      endif
    endelse
  endfor

  allid = ulong64(total(cat_id,/double) + cat_idlast)
  allarea = ulong64(total(cat_area,/double) + cat_arealast)
  ss = moment(oba, sdev=sdev, mean=aps) & medps = long64(median(oba))

  ;; ext3 = nw_area per ID in extended size
  ext3=long(0)*temporary(nw_ids)
  ;; assign object area to object polygon
  nr_obj = n_elements(obj_area)
  FOR i = 1l, nr_obj - 1 DO ext3[revind[revind[i]:revind[i + 1] - 1]] = obj_area(i)
  obj_area_saved = obj_area  ;; replaces: save, obj_area, filename = info.dir_tmp + 'obj_area.sav'

  ;; ext = viewport image, assign colors for up to 6 area classes
  for i_class = 0, nr_cat do begin
    if i_class eq nr_cat then begin
      qsmall = where(obj_area NE 9000000000000000000,c_small, /l64)
    endif else begin
      qsmall = where(obj_area LE cat(i_class),c_small, /l64) ;; where the histogram area is in area class
    endelse
    if c_small gt 0 then begin
      for id = 0, c_small-1 do begin
        il = qsmall[id] & q = revind[revind[il]:revind[il+1]-1] & ext[q] = cl(i_class)
      endfor
      ;; set qsmall to a high number
      if i_class eq nr_cat then obj_area = 0 else obj_area[qsmall] = 9000000000000000000
    endif
  endfor

  id_max1 = -1 & id_max2 = -1 & id_max3 = -1 & oba_max1 = 0 & oba_max2 = 0 & oba_max3 = 0
  ;; show the 3 largest objects in pink (80b)
  if nr_obj gt 3 and (big3pink eq 1) then begin
    obj_area = obj_area_saved  ;; replaces: restore, info.dir_tmp + 'obj_area.sav'
    oba_max1 = oba_max & id_max1 = 0 & oba_max2 = 0 & id_max2 = 0 & oba_max3 = 0 & id_max3 = 0
    ;; the biggest object
    qsmall = where(obj_area eq oba_max1, c_small, /l64)
    if c_small gt 0 then begin
      for id = 0, c_small-1 do begin
        il = qsmall[id] & q = revind[revind[il]:revind[il+1]-1] & ext[q] = 80b
      endfor
      ;; set qsmall to a high number
      obj_area[qsmall] = 9000000000000000000
      id_max1 = qsmall
    endif
    ;; the second biggest object
    oba_max2 = max((obj_area lt 9000000000000000000)*obj_area)
    qsmall = where(obj_area eq oba_max2, c_small, /l64)
    if c_small gt 0 then begin
      for id = 0, c_small-1 do begin
        il = qsmall[id] & q = revind[revind[il]:revind[il+1]-1] & ext[q] = 80b
      endfor
      ;; set qsmall to a high number
      obj_area[qsmall] = 9000000000000000000
      id_max2 = qsmall
    endif
    ;; the third biggest object
    oba_max3 = max((obj_area lt 9000000000000000000)*obj_area)
    qsmall = where(obj_area eq oba_max3, c_small, /l64)
    if c_small gt 0 then begin
      for id = 0, c_small-1 do begin
        il = qsmall[id] & q = revind[revind[il]:revind[il+1]-1] & ext[q] = 80b
      endfor
      ;; set qsmall to a high number
      obj_area[qsmall] = 9000000000000000000
      id_max3 = qsmall
    endif
  endif

  ;; cut back to original
  ext = temporary(ext[5:sz[0]+4,5:sz[1]+4])
  ext3 = temporary(ext3[5:sz[0]+4,5:sz[1]+4])

  ;; put back missing
  q=where(im eq 0b, ct_q, /l64)
  IF ct_q GT 0 THEN BEGIN
    ext(q) = 129b ; image in viewport
    ext2(q)= -1   ; id of object
    ext3(q)= -1   ; area of object
  ENDIF
  q = where(im eq 3b, ct_q, /l64) ;; water bodies - blue
  IF ct_q GT 0 THEN BEGIN
    ext(q) = 105b ; image in viewport
    ext2(q)= -3   ; id of object
    ext3(q)= -3   ; area of object
  ENDIF
  q = where(im eq 4b, ct_q, /l64) ;; special BG -pale-blue
  IF ct_q GT 0 THEN BEGIN
    ext(q) = 176b ; image in viewport
    ext2(q)= -4   ; id of object
    ext3(q)= -4   ; area of object
  ENDIF

  ;; first rotate the images back to normal (as written by GTB)
  openw, u, outdir + 'classes.raw', /get_lun & writeu, u, rotate(ext, 7) & free_lun, u
  openw, u, outdir + 'ids.raw', /get_lun & writeu, u, long(rotate(ext2, 7)) & free_lun, u
  openw, u, outdir + 'pixels.raw', /get_lun & writeu, u, long(rotate(ext3, 7)) & free_lun, u

  ;; the numbers of the txt/csv reports, formatted as GTB prints them
  openw, u, outdir + 'stats.txt', /get_lun
  printf, u, 'nr_cat=' + strtrim(nr_cat,2)
  printf, u, 'cat=' + strjoin(strtrim(cat,2), ',')
  printf, u, 'cat_id=' + strjoin(strtrim(cat_id,2), ',') + ',' + strtrim(cat_idlast,2)
  printf, u, 'cat_area=' + strjoin(strtrim(cat_area,2), ',') + ',' + strtrim(cat_arealast,2)
  printf, u, 'pct_objects=' + strjoin(strtrim(cat_id*100.0/tot_ids,2), ',') + ',' + strtrim(cat_idlast*100.0/tot_ids,2)
  printf, u, 'pct_area=' + strjoin(strtrim(cat_area*100.0/tot_area,2), ',') + ',' + strtrim(cat_arealast*100.0/tot_area,2)
  printf, u, 'tot_ids=' + strtrim(tot_ids,2)
  printf, u, 'tot_area=' + strtrim(tot_area,2)
  printf, u, 'allid=' + strtrim(allid,2)
  printf, u, 'allarea=' + strtrim(allarea,2)
  printf, u, 'median=' + strtrim(medps,2)
  printf, u, 'mean=' + strtrim(aps,2)
  printf, u, 'sdev=' + strtrim(sdev,2)
  printf, u, 'oba_max=' + strtrim(oba_max,2)
  printf, u, 'id_max1=' + strjoin(strtrim(id_max1,2), ',')
  printf, u, 'id_max2=' + strjoin(strtrim(id_max2,2), ',')
  printf, u, 'id_max3=' + strjoin(strtrim(id_max3,2), ',')
  printf, u, 'oba_max1=' + strtrim(oba_max1,2)
  printf, u, 'oba_max2=' + strtrim(oba_max2,2)
  printf, u, 'oba_max3=' + strtrim(oba_max3,2)
  free_lun, u
end
