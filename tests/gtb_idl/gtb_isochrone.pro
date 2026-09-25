; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Restoration Planner -> Isochrone Map A / AB"
; for the landschaft test-suite: the statements of the 'cost_map' event of
; guidostoolbox.pro (lines 17353-17510) copied verbatim, except for the
; progress bar and dialog lines. They run GTB's ggeo program from
; <dir_tmp>/../spatcon/ggeo64.exe.
;
;   gtb_isochrone, 'resist.tif', 'marker.tif', 'dir_tmp/', 'out.raw', mode   (mode 'cost_map_aa' or 'cost_map_ab')
;
; resist.tif is read and flipped like GTB's current image (rotate(.., 7)); the
; marker map is read as GTB reads it. out.raw: the byte isochrone map in the row
; order of the input files.

pro gtb_isochrone, resist_file, marker_file, dir_tmp, out_file, eventvalue2
  info = {dir_tmp: dir_tmp, my_os: 'windows', add_title: '', log: dir_tmp + 'events.txt', bline: ''}
  resist = rotate(read_tiff(resist_file), 7)
  marker = read_tiff(marker_file)

      ;; setup and calculate cost map
      ;;================================================================================
      pushd, info.dir_tmp
      list = file_search() & nl = n_elements(list)
      IF list[0] NE '' THEN FOR i = 0, nl -1 DO file_delete, list[i], / allow_nonexistent, / quiet, / recursive
      
      sz = size(marker) & eew = 2  & eew2 = eew * 2

      ;; put stuff in place
      ;; use enlarged images to avoid getting the zero line at the image boundaries
      ;; get missing data in original resolution
      qm = where(rotate(marker,7) eq 3b, ctqm)
      marktmp = marker 
      q2b = where(marker eq 2b,ct_q2b)
      if ct_q2b gt 0 then marktmp[q2b] = 0b      
      ;; marker1 with extended frame of missing (3b)
      ext = bytarr(sz[1] + eew2, sz[2] + eew2) + 3b
      ext[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1] = marktmp      
      write_tiff,'marker1.tif',rotate(ext,7),compression=1
            
      ;; reassign resistance image: FG-1 for cost analysis
;;;      q = where(resist eq 2b) & resist[q] = 1b & q = 0
;;;      ext = bytarr(sz[1] + eew2, sz[2] + eew2)
;;;      ext[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1] = resist   
;;;      write_tiff,'mask.tif', rotate(ext,7) ,compression=1

      ;; reassign resistance image for cost analysis: FG-1 
      ;; write out resistance mask with extended frame of 0b
      q = where(resist eq 2b) & resist[q] = 1b & q = 0
      ext = bytarr(sz[1] + eew2, sz[2] + eew2)
      ext[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1] = resist
      write_tiff,'mask.tif', ext, compression=1
      
;      resisttmp = rotate(resist,7)
;      write_tiff,'mask.tif',resisttmp,compression=1

      IF info.my_os EQ 'windows' THEN BEGIN
        IF (!version.memory_bits EQ 64) THEN ggeo='..\spatcon\ggeo64.exe' ELSE $
          ggeo='..\spatcon\ggeo32.exe'
        file_copy, ggeo, 'ggeo.exe', /overwrite
      ENDIF ELSE IF info.my_os EQ 'darwin' THEN BEGIN
        ggeo='../spatcon/ggeo_mac' & file_copy, ggeo, 'ggeo', /overwrite
      ENDIF ELSE IF info.my_os EQ 'apple' THEN BEGIN
        ggeo='../spatcon/ggeoARM_mac' & file_copy, ggeo, 'ggeo', /overwrite
      ENDIF ELSE BEGIN
        ggeo='../spatcon/ggeo_lin64' & file_copy, ggeo, 'ggeo', /overwrite
      ENDELSE


      case eventvalue2 of
        'cost_map_aa': begin ;; [0, 1]
          widget_control, / hourglass

          ;; run ggeo in tmp
          IF info.my_os EQ 'windows' THEN spawn, 'ggeo.exe', log, / hide ELSE spawn, './ggeo', log

          cost = read_tiff('out.tif')
          cost = cost[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1]
          ;; long64:
          ;; mincost = long64((min(cost-1) + 1)/2.0)
          ;; cost = round(temporary(cost)/2.0,/l64)
          ;; long32
          mincost = round((min(cost-1) + 1)/2.0)
          cost = round(temporary(cost)/2.0)
          maxcost=max(cost)
          ;; the actual cost map
          marker=rotate(marker,7)
          qmarker = where(marker eq 1b)
          if ctqm gt 0 then maxcost=max((cost lt maxcost)*cost)
          ;; test if image is too large for long 32 processing
          if maxcost lt 0 then begin
            msg = 'Map is too large for current implementation.' + string(10b) + 'Returning.'
            openu, unit, info.log,/append, /Get_lun & printf, unit, systime() + ', ERROR: ' + msg + info.bline & free_lun, unit
            GOTO, fin
          endif

          ;; test for unreachable pixels, i.e., a river cutting off part of the image
          q = where(cost lt -2, ct, /l64)
          ;; set up cost image for display, range 0-100 cost, objects 101, missing 102, lcp 103
          cost = BytScl(cost, min=0, max=maxcost, Top = 100)
          if ct gt 0 then cost[q] = 104b ;; unreachable
          cost[qmarker] = 101b & if ctqm gt 0 then cost[qm] = 102b
          info.add_title = ' (isochrone map A)'
        end

        'cost_map_ab': begin ;; [0, 1, 2]
          widget_control, / hourglass
          marktmp = marker
          q1b = where(marktmp eq 1b,ct_q1b, /l64)
          if ct_q1b gt 0 then marktmp(q1b) = 0b
          if ct_q2b gt 0 then marktmp(q2b) = 1b
          ;; marker2 with extended frame of missing (3b)
          ext = bytarr(sz[1] + eew2, sz[2] + eew2) + 3b
          ext[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1] = marktmp         
          write_tiff,'marker2.tif',rotate(ext,7),compression=1
          
          marker = (marker - 1b) lt 2b  ;; marker mask without missing data of 3b
          ext = bytarr(sz[1] + eew2, sz[2] + eew2)
          ext[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1] = marker
          write_tiff,'markers.tif',rotate(ext,7),compression=1


          IF info.my_os EQ 'windows' THEN spawn, 'ggeo.exe', log, / hide ELSE spawn, './ggeo', log
          ;; cost of A, B, read and  set unreachable pixels to zero
          cost = read_tiff('out.tif')
          cost = cost[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1]
          ;; long64:
          ;; cost = round(temporary(cost)/2.0,/l64)
          ;; maxcost = max(cost)
          ;; if ct_miss gt 0 then begin
          ;;    maxcost=max((cost lt maxcost)*cost)
          ;; endif
          ;; mincost = min(cost)

          ;; long 32
          mincost = round(min(cost)/2.0)
          maxcost = max(cost)
          if ctqm gt 0 then maxcost=round((max((cost lt maxcost)*cost))/2.0) else maxcost=round(maxcost/2.0)

          ;; test if image is too large for long 32 processing
          if maxcost lt 0 then begin
            msg = 'Map is too large for current implementation.' + string(10b) + 'Returning.'
            openu, unit, info.log,/append, /Get_lun & printf, unit, systime() + ', ERROR: ' + msg + info.bline & free_lun, unit
            GOTO, fin
          endif
          ;; convert ulong to long 32
          cost = round(temporary(cost)/2.0)
          ;; end long 32

          ;; get markers and least cost path
          marker=rotate(marker,7)
          qmarker = where(marker eq 1b, /l64)
          
          ;; get the lcp, on apple we need to convert it from the pgm workaround
          lcp = read_tiff('lcp.tif')                 
          lcp = lcp[eew:eew + sz[1] - 1, eew:eew + sz[2] - 1]
          lcp = where(lcp gt 0b, ct_lcp, /l64)

          ;; test for unreachable pixels, i.e., a river cutting off part of the image
          ;; for long 64: q = where(cost gt maxcost, ct)
          q = where(cost lt -2, ct, /l64)
          ;; set up cost image for display, range 0-100 cost, objects 101, missing 102, lcp 103
          cost = BytScl(cost, min = mincost, max = maxcost, Top = 100)
          cost[lcp] = 103b & cost[qmarker] = 101b & if ctqm gt 0 then cost[qm]= 102b
          if ct gt 0 then cost[q] = 104b ;; unreachable
          info.add_title = ' (isochrone map AB)'
        end
      endcase
      popd
  fin:
  openw, u, out_file, /get_lun & writeu, u, rotate(cost, 7) & free_lun, u
end
