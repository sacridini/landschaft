; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Image Analysis -> Pattern -> Morphological ->
; MSPA Tiling" for the landschaft test-suite: the processing statements of the
; 'mspatile' event of guidostoolbox.pro (lines 11791-12247) copied verbatim,
; except for the progress bar, dialog and cancel lines. The tile size (GTB:
; 6500 on MS-Windows, 9500 on Linux, 8500 on macOS) and the overlay (2000) are
; parameters, so that small test maps span many tiles. It runs GTB's gdal
; tools (<fwtools>/setfw.bat) and <root>/mspa.exe; tiles go to <root>/mspatmp/.
;
;   gtb_mspa_tile, 'dir/in.tif', 'root/', 'C:\GuidosToolbox\guidos_progs\GTBtools\', '8', '1', '1', '1', 6500, 2000

pro gtb_mspa_tile, fname, root, fwtools, c_FGconn, c_size, c_trans, c_intext, stepx_in, overlay_in
  info = {dir_tmp: root + 'mspatmp/', dir_guidossub: root, dir_fwtools: fwtools, $
    windrive: strmid(fwtools, 0, 2), my_os: 'windows', sysgdal: '', log: root + 'log.txt', bline: ''}
  dir_dat = file_dirname(fname, / mark)
  res = query_tiff(fname, inpinfo, geotiff = geotiffx)
  IF c_trans EQ '1' THEN restore, info.dir_guidossub + 'mspacolorston.sav' ELSE $
    restore, info.dir_guidossub + 'mspacolorstoff.sav'

      ;; now we are ready for processing
      time0 = systime( / sec)

      ;;====================================================================
      ;; define gdal commands, overlay, olt-stepsize
      ;;====================================================================
      ;; gdal_translate, buildvrt
      IF info.my_os EQ 'windows' THEN BEGIN
         gtrans = info.windrive + ' & cd "' + info.dir_fwtools + '" & setfw.bat & ' + 'cd "' + $
          info.dir_tmp + '" & gdal_translate -co "COMPRESS=LZW" -srcwin '
         gbuildvrt = info.windrive + ' & cd "' + info.dir_fwtools + '" & setfw.bat & ' + 'cd "' + $
          info.dir_tmp + '" & gdalbuildvrt -input_file_list '
      ENDIF ELSE BEGIN ;; linux, osx
         if strlen(info.sysgdal) gt 0 then gtrans = 'unset LD_LIBRARY_PATH; gdal_translate -co "COMPRESS=LZW" -srcwin ' else $
          gtrans = info.dir_fwtools + 'gdal_translate -co "COMPRESS=LZW" -srcwin '
         if strlen(info.sysgdal) gt 0 then gbuildvrt = 'unset LD_LIBRARY_PATH; gdalbuildvrt ' else $
          gbuildvrt = info.dir_fwtools + 'gdalbuildvrt '
      ENDELSE      
          
      ;;[-srcwin xoff yoff xsize ysize] and yoffset is counted from top
      ;; the entire image is xsize, ysize
      ;; sub-tile into tiles of 10000x10000 with an overlay of 2000 so the
      ;; final adjacent tiles will be of size 8000x8000
      ;; - stepx/y: tilesize including the overlay area
      ;; - dx/y: tilesize without overlay (8000x8000)
      ;; - nrtx/y = number of tiles in x and y

      ;; dimensions in x and y
      xtot = inpinfo.dimensions[0] & ytot = inpinfo.dimensions[1]
      overlay = long(overlay_in)  ;; shift of common overlay

      ;; settings for the olts and the final tiles to merge back
      stepx = long(stepx_in) ;; GTB: 6500 for MS-Windows
      stepy = stepx
      dx = stepx - overlay & dy = dx ;; actual stepsize of olts
      ;; border to remove from overlays
      border = overlay / 2 & sborder = ' ' + strtrim(border, 2)
      ;; dimension of merger tiles
      mex = ' ' + strtrim(stepx - overlay, 2) & mey = mex

      ;; mspa parameters
      suffix = '_' + strtrim(c_FGconn, 2) + '_' + strtrim(c_size, 2) + '_' + $
               strtrim(c_trans, 2) + '_' + strtrim(c_intext, 2) + '.tif'
      ;;=================================================================
      ;; Part 1: automatic loop of cutting overlays for MSPA
      ;;=================================================================
      ;; inform

      ;; start with an empty tmp folder
      pushd, info.dir_tmp
      list = file_search() & nl = n_elements(list)
      IF list[0] NE '' THEN FOR i = 0, nl - 1 DO file_delete, list[i], / allow_nonexistent, / quiet, / recursive

      iy = - 1 & lasty = 0
      yloop:
      iy = iy + 1

      ix = - 1
      xloop:
      ix = ix + 1

      ;; start/end points of overlay tile
      xx1 = strtrim(ix * dx) & xx2 = ix * dx + stepx
      yy1 = strtrim(iy * dy) & yy2 = iy * dy + stepy
      ddx = strtrim(stepx) & ddy = strtrim(stepy)
      subt = 'olt_' + strtrim(ix + 1, 2) + '_' + strtrim(iy + 1, 2) + '.tif'

      ;; check for last tiles in Y which is not necessarily full size
      IF yy2 GE ytot THEN BEGIN ;; we are in the last tile in Y
         ddy = strtrim(ytot - yy1) & lasty = 1
      ENDIF

      ;; check for last tile in X which is not necessarily full size
      IF xx2 GE xtot THEN BEGIN ;; we are in the last tile in X
         ddx = strtrim(xtot - xx1)
         cmd = gtrans + xx1 + yy1 + ddx + ddy + ' ' + fname + ' ' + subt
         IF info.my_os EQ 'windows' THEN $
          spawn, cmd, log, / hide ELSE spawn, cmd, log
         IF lasty EQ 0 THEN GOTO, yloop ELSE GOTO, olayend
      ENDIF

      ;; cut the overlays
      cmd = gtrans + xx1 + yy1 + ddx + ddy + ' "' + fname + '" ' + subt
      IF info.my_os EQ 'windows' THEN $
       spawn, cmd, log, / hide ELSE spawn, cmd, log
      GOTO, xloop

      olayend:

      ;; list of tiles in x and y for part 3
      list_oltx = file_search('olt_*_1.tif', count = nr_oltx)
      list_olty = file_search('olt_1_*.tif', count = nr_olty)
      popd ;; back to the roots
      ;;=================================================================
      ;; Part 1 finished: cutting olay tiles
      ;;=================================================================

      part2:
      ;;=================================================================
      ;; Part 2: do mspa
      ;;=================================================================
      ;; get the list of tiles for MSPA processing
      list_olt = $
       file_search(info.dir_tmp + 'olt*.tif', count = nr_olt)



      FOR isub = 0, nr_olt - 1 DO BEGIN      

         fbase0 = file_basename(list_olt(isub))
         fbase1 = 'out' + strmid(fbase0, 3)

         ;; test if MSPA processing can be skipped
         ;;========================================       
         ;; gdal statistics will not calculate min/max if only nodata is present!   
         ;; since in that case gdalinfo -mm does not work we have to read the image 
         res = read_tiff(list_olt(isub)) & mxx=max(res,min=mii) & res = 0
         
         IF mxx EQ 0b THEN BEGIN
            ;; the entire olt is Nodata [0]
            pushd, info.dir_tmp
            res = query_tiff(fbase0, inpinfo, geotiff = geotiffx)
            imtmp = bytarr(inpinfo.dimensions[0], inpinfo.dimensions[1]) + 129b
            write_tiff, fbase1, imtmp, geotiff = geotiffx, $
                        compression = 1, red = r, green = g, blue = b
            imtmp = 0 & popd & GOTO, skipmsapt
         ENDIF
         
         IF mii EQ 2b THEN BEGIN
            ;; the entire olt is FG [2]
            pushd, info.dir_tmp
            res = query_tiff(fbase0, inpinfo, geotiff = geotiffx)
            imtmp = bytarr(inpinfo.dimensions[0], inpinfo.dimensions[1]) + 17b
            write_tiff, fbase1, imtmp, geotiff = geotiffx, $
                        compression = 1, red = r, green = g, blue = b
            imtmp = 0 & popd & GOTO, skipmsapt
         ENDIF

         IF mxx EQ 1b AND mii LE 1b THEN BEGIN
            ;; either the entire olt is BG [1] or a part of it
            pushd, info.dir_tmp
            imtmp = read_tiff(fbase0, geotiff = geotiffx)
            imtmp = (imtmp EQ 0b) * 129b           
            write_tiff, fbase1, imtmp, geotiff = geotiffx, $
                        compression = 1, red = r, green = g, blue = b
            imtmp = 0 & popd & GOTO, skipmsapt
         ENDIF

         imtmp = read_tiff(list_olt(isub), geotiff = geotiffx)
         qq = where(imtmp EQ 1b, nr_qq, /l64) & qq = 0
         
         IF nr_qq EQ 0 THEN BEGIN
            ;; there is only Nodata and FG [0,2]
            pushd, info.dir_tmp
            imtmp = (imtmp EQ 2b) * 17b + (imtmp EQ 0b) * 129b
            write_tiff, fbase1, imtmp, geotiff = geotiffx, $
                        compression = 1, red = r, green = g, blue = b
            imtmp = 0 & popd & GOTO, skipmsapt
         ENDIF
         IF info.my_os EQ 'windows' THEN BEGIN
            ;; all 3 or BG/FG: normal MSPA processing [0,1,2] or [1,2]
            pushd, info.dir_guidossub
            cmd = 'mspa.exe -graphfg ' + c_FGconn + $
                  ' -eew ' + c_size + ' -internal ' + c_intext + $
                  ' -disk -transition ' + c_trans + $
                  ' -i mspatmp\' + fbase0 + $
                  ' -o ' + fbase1 + ' -odir mspatmp\'
             spawn, cmd, log, / hide
            popd
         ENDIF ELSE BEGIN
            pushd, info.dir_tmp
            IF info.my_os EQ 'linux' THEN mspa_os = 'mspa_lin' 
            IF info.my_os EQ 'darwin' THEN mspa_os = 'mspa_mac'
            IF info.my_os EQ 'apple' THEN mspa_os = 'mspa_mac'
            cmd = info.dir_guidossub + mspa_os + ' -graphfg ' + c_FGconn + $
                  ' -eew ' + c_size + ' -internal ' + c_intext + $
                  ' -disk -transition ' + c_trans + $
                  ' -i ' + fbase0 + ' -o ' + fbase1 + ' -odir ./'
             spawn, cmd, log
            popd
         ENDELSE
         ;; get the result and add colortable and geotiff info
         pushd, info.dir_tmp
         res = query_tiff(fbase0, inpinfo, geotiff = geotiffx)
         imtmp = read_tiff(fbase1)
         write_tiff, fbase1, imtmp, geotiff = geotiffx, $
                     compression = 1, red = r, green = g, blue = b
         imtmp = 0
         popd

         skipmsapt:
      ENDFOR
      


      ;;=================================================================
      ;; Part 2 finished: do mspa
      ;;=================================================================

      ;;=================================================================
      ;; Part 3: remove overlay and merge back to one image
      ;;=================================================================
      ;; we can have 3 options denoted in [columns, rows]
      ;; 1) long single row [n, 1] or long single column [1, n]; n ge 2
      ;; 2) long double row [n, 2] or long double column [2, n]; n ge 2
      ;; 3) with center tile(s) [m, n]; m and/or n gt 2
      ;;
      pushd, info.dir_tmp

      ;; case 1a) one long single row
      ;;----------------------------------
      IF nr_olty EQ 1 THEN BEGIN
         ;; leftmost olt
         fn_in = 'out_1_1.tif'
         fn_out = ' res_1_1.tif'
         cmd = gtrans + '0 0 ' + strtrim(mex + border, 2) + ' ' + $
               strtrim(ytot, 2) + ' ' + fn_in + fn_out
         IF info.my_os EQ 'windows' THEN $
          spawn, cmd, log, / hide ELSE spawn, cmd, log

         ;; rightmost olt
         fn_in = 'out_' + strtrim(nr_oltx, 2) + '_1.tif'
         fn_out = ' res_' + strtrim(nr_oltx, 2) + '_1.tif'
         q = query_tiff(fn_in, infox) & sztr = infox.dimensions
         cmd = gtrans + sborder + ' 0 ' + strtrim(sztr(0) - border, 2) + $
               ' ' + strtrim(ytot, 2) + ' ' + fn_in + fn_out
         IF info.my_os EQ 'windows' THEN $
          spawn, cmd, log, / hide ELSE spawn, cmd, log

         ;; center olts: we need to cut tile 2 to nr_oltx - 1 olt-tiles
         IF nr_oltx GT 2 THEN BEGIN
            FOR idx = 2, nr_oltx - 1 DO BEGIN
               fn_in = 'out_' + strtrim(idx, 2) + '_1.tif'
               fn_out = ' res_' + strtrim(idx, 2) + '_1.tif'
               cmd = gtrans + sborder + ' 0' + mex + ' ' + $
                     strtrim(ytot, 2) + ' ' + fn_in + fn_out
               IF info.my_os EQ 'windows' THEN $
                spawn, cmd, log, / hide ELSE spawn, cmd, log
            ENDFOR
         ENDIF
         GOTO, finalmerge
      ENDIF


      ;; case 1b) one long single column
      ;;----------------------------------
      IF nr_oltx EQ 1 THEN BEGIN
         ;; top olt
         fn_in = 'out_1_1.tif'
         fn_out = ' res_1_1.tif'
         cmd = gtrans + '0 0 ' + strtrim(xtot, 2) + ' ' + $
               strtrim(mey + border, 2) + ' ' + fn_in + fn_out
         IF info.my_os EQ 'windows' THEN $
          spawn, cmd, log, / hide ELSE spawn, cmd, log

         ;; bottom olt
         fn_in = 'out_1_' + strtrim(nr_olty, 2) + '.tif'
         fn_out = ' res_1_' + strtrim(nr_olty, 2) + '.tif'
         q = query_tiff(fn_in, infox) & szbl = infox.dimensions
         cmd = gtrans + '0' + sborder + ' ' + strtrim(xtot, 2) + ' ' + $
               strtrim(szbl(1) - border, 2) + ' ' + fn_in + fn_out
         IF info.my_os EQ 'windows' THEN $
          spawn, cmd, log, / hide ELSE spawn, cmd, log

         ;; center olts: we need to cut tile 2 to nr_olty - 1 olt-tiles
         IF nr_olty GT 2 THEN BEGIN
            FOR idy = 2, nr_olty - 1 DO BEGIN
               fn_in = 'out_1_' + strtrim(idy, 2) + '.tif'
               fn_out = ' res_1_' + strtrim(idy, 2) + '.tif'
               cmd = gtrans + '0' + sborder + ' ' + strtrim(xtot, 2) + $
                     mey + ' ' + fn_in + fn_out
               IF info.my_os EQ 'windows' THEN $
                spawn, cmd, log, / hide ELSE spawn, cmd, log
            ENDFOR
         ENDIF
         GOTO, finalmerge
      ENDIF

      ;; case 2)
      ;;----------------------------------
      ;; if we arrive here we have at least a 2x2 tiling

      ;; 1) cut the corner olts only
      ;;----------------------------------
      ;; top left: 1/1
      fn_in = 'out_1_1.tif'
      fn_out = ' res_1_1.tif'
      cmd = gtrans + '0 0 ' + strtrim(mex + border, 2) + ' ' + $
            strtrim(mey + border, 2) + ' ' + fn_in + fn_out
      IF info.my_os EQ 'windows' THEN $
       spawn, cmd, log, / hide ELSE spawn, cmd, log

      ;; top right: nr_oltx/1
      fn_in = 'out_' + strtrim(nr_oltx, 2) + '_1.tif'
      fn_out = ' res_' + strtrim(nr_oltx, 2) + '_1.tif'
      q = query_tiff(fn_in, infox) & sztr = infox.dimensions
      cmd = gtrans + sborder + ' 0 ' + strtrim(sztr(0) - border, 2) + $
            ' ' + strtrim(mey + border, 2) + ' ' + fn_in + fn_out
      IF info.my_os EQ 'windows' THEN $
       spawn, cmd, log, / hide ELSE spawn, cmd, log

      ;; bottom left: 1/nr_olty
      fn_in = 'out_1_' + strtrim(nr_olty, 2) + '.tif'
      fn_out = ' res_1_' + strtrim(nr_olty, 2) + '.tif'
      q = query_tiff(fn_in, infox) & szbl = infox.dimensions
      cmd = gtrans + '0' + sborder + ' ' + strtrim(mex + border, 2) + ' ' + $
            strtrim(szbl(1) - border, 2) + ' ' + fn_in + fn_out
      IF info.my_os EQ 'windows' THEN $
       spawn, cmd, log, / hide ELSE spawn, cmd, log

      ;; bottom right: nr_oltx/nr_olty
      fn_in = 'out_' + strtrim(nr_oltx, 2) + '_' + strtrim(nr_olty, 2) + '.tif'
      fn_out = ' res_' + strtrim(nr_oltx, 2) + '_' + $
               strtrim(nr_olty, 2) + '.tif'
      q = query_tiff(fn_in, infox) & szbr = infox.dimensions
      cmd = gtrans + sborder + sborder + ' ' + $
            strtrim(szbr(0) - border, 2) + ' ' + $
            strtrim(szbr(1) - border, 2) + ' ' + fn_in + fn_out
      IF info.my_os EQ 'windows' THEN $
       spawn, cmd, log, / hide ELSE spawn, cmd, log

      ;; so far we are ok for a 2x2, now check for more olts
      ;;  cut the boundary frame olts only
      ;;----------------------------------
      IF nr_oltx GT 2 THEN BEGIN ;; cut top and bottom row
         ;; top row: we need to cut tile 2 to nr_oltx - 1 olt-tiles
         FOR idx = 2, nr_oltx - 1 DO BEGIN
            fn_in = 'out_' + strtrim(idx, 2) + '_1.tif'
            fn_out = ' res_' + strtrim(idx, 2) + '_1.tif'
            cmd = gtrans + sborder + ' 0' + mex + ' ' + $
                  strtrim(mey + border, 2) + ' ' + fn_in + fn_out
            IF info.my_os EQ 'windows' THEN $
             spawn, cmd, log, / hide ELSE spawn, cmd, log
         ENDFOR
         ;; bottom row: we need to cut nr_oltx - 2 olt-tiles
         FOR idx = 2, nr_oltx - 1 DO BEGIN
            str2 = strtrim(idx, 2) + '_' + strtrim(nr_olty, 2)
            fn_in = 'out_' + str2 + '.tif'
            fn_out = ' res_' + str2 + '.tif'
            cmd = gtrans + sborder + sborder + mex + ' ' + $
                  strtrim(szbr(1) - border, 2) + ' ' + fn_in + fn_out
            IF info.my_os EQ 'windows' THEN $
             spawn, cmd, log, / hide ELSE spawn, cmd, log
         ENDFOR
      ENDIF

      IF nr_olty GT 2 THEN BEGIN ;; cut left and right column
         ;; left column: we need to cut nr_olty - 2 olt-tiles
         FOR idy = 2, nr_olty - 1 DO BEGIN
            fn_in = 'out_1_' + strtrim(idy, 2) + '.tif'
            fn_out = ' res_1_' + strtrim(idy, 2) + '.tif'
            cmd = gtrans + '0' + sborder + ' ' + strtrim(mex + border, 2) + $
                  mey + ' ' + fn_in + fn_out
            IF info.my_os EQ 'windows' THEN $
             spawn, cmd, log, / hide ELSE spawn, cmd, log
         ENDFOR
         ;; right column: we need to cut nr_olty - 2 olt-tiles
         FOR idy = 2, nr_olty - 1 DO BEGIN
            str2 = strtrim(nr_oltx, 2) + '_' + strtrim(idy, 2)
            fn_in = 'out_' + str2 + '.tif'
            fn_out = ' res_' + str2 + '.tif'
            cmd = gtrans + sborder + sborder + ' ' + $
                  strtrim(szbr(0) - border, 2) + mey + ' ' + fn_in + fn_out
            IF info.my_os EQ 'windows' THEN $
             spawn, cmd, log, / hide ELSE spawn, cmd, log
         ENDFOR
      ENDIF

      IF nr_oltx GT 2 AND nr_olty GT 2 THEN BEGIN
         ;; cut the remaining 'normal' olts inside the frame
         FOR idy = 2, nr_olty - 1 DO BEGIN
            FOR idx = 2, nr_oltx - 1 DO BEGIN
               str2 = strtrim(idx, 2) + '_' + strtrim(idy, 2)
               fn_in = 'out_' + str2 + '.tif'
               fn_out = ' res_' + str2 + '.tif'
               cmd = gtrans + sborder + sborder + mex + mey + $
                     ' ' + fn_in + fn_out
               IF info.my_os EQ 'windows' THEN $
                spawn, cmd, log, / hide ELSE spawn, cmd, log
            ENDFOR
         ENDFOR
      ENDIF

      finalmerge:
      popd ;; leave tmp
      ;;=================================================================
      ;; removal of overlays finished
      ;;=================================================================
      ;; go up to dir_dat and merge into one big image
      zz = file_basename(fname) & zz = strmid(zz,0,strpos(zz,'.',/reverse_search))
      fn_out = dir_dat + zz + suffix
      ;; the MSPA settings watermark
      wm = ' ' + strmid(file_basename(suffix,'.tif'),1)

      ;;===================================
      IF info.my_os EQ 'windows' THEN BEGIN
         pushd, info.dir_tmp
	       qq = file_search('res*.tif')
	       openw, 1, 'reslist.txt' & for i = 0l, n_elements(qq)-1 do printf,1,qq[i] & close,1
	       ;; setup vrt image
	       spawn, gbuildvrt + 'reslist.txt xx.vrt', /hide
	       ;; cmd for watermark and merging	       
	       cmd = info.windrive + ' & cd "' + info.dir_fwtools + '" & setfw.bat & cd "' + $
	         info.dir_tmp + '" & gdal_translate -mo ' + $
	         '"TIFFTAG_IMAGEDESCRIPTION=GTB_MSPA, https://forest.jrc.ec.europa.eu/activities/lpa/gtb/' + $
	         wm + '" -co "COMPRESS=LZW" xx.vrt "' + fn_out + '"' 
         popd
      ;;=================================
      ENDIF ELSE BEGIN
         ;; OSX and Linux
         ;; setup vrt image
         spawn, gbuildvrt + info.dir_tmp + 'xx.vrt ' + info.dir_tmp + 'res*.tif'             
         ;; cmd for watermark and merging  
         if strlen(info.sysgdal) gt 0 then cmd = 'unset LD_LIBRARY_PATH; gdal_translate -mo ' else $
          cmd = info.dir_fwtools + 'gdal_translate -mo '
         cmd = cmd + '"TIFFTAG_IMAGEDESCRIPTION=GTB_MSPA, https://forest.jrc.ec.europa.eu/activities/lpa/gtb/' + $
          wm + '" -co "COMPRESS=LZW" "' + info.dir_tmp + 'xx.vrt" "' + fn_out + '"'
      ENDELSE
      IF info.my_os EQ 'windows' THEN spawn, cmd, log, / hide ELSE spawn, cmd, log


      ;; remove the intermediate files
      pushd, info.dir_tmp
      list = file_search() & nl = n_elements(list)
      IF list[0] NE '' THEN FOR i = 0, nl -1 DO file_delete, list[i], / allow_nonexistent, / quiet, / recursive
      popd ;; back to the roots
      ;;=================================================================
      ;; Part 3 finished: remove overlay and merge back to one image
      ;;=================================================================

      fin:
end
