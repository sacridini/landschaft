; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Pattern -> Morphological -> SPA2/3/5/6" for
; the landschaft test-suite: the processing statements of the 'spa' event of
; guidostoolbox.pro (lines 9104-9266) copied verbatim. The input is
; read and flipped like GTB's current image; the statistics are written to a
; key=value file instead of the SPA report.
;
;   gtb_spa, 'in.tif', 'out/', 'spa5'

pro gtb_spa, input, outdir, eventValue2
  image0 = rotate(read_tiff(input), 7)
     se8 = replicate(1b, 3, 3) ;; 8-conn kernel
     ;; check for missing  pixels
     image0min = min(image0) & ct_qmiss = 0
     if image0min eq 0b then begin
       missing = image0 eq 0b 
       qmiss = where(missing eq 1b, ct_qmiss, /l64)      
     endif
     area_image = (size(image0))[4]
     area_data = area_image - ct_qmiss
     
     ;; extend the working image to ensure proper results for label_region
     sz = size(image0,/dim) & ext1 = bytarr(sz(0) + 4, sz(1) + 4) + 1b & sz2=size(ext1,/dim) 
     ext1[2:sz[0]+1, 2:sz[1]+1] = temporary(image0) & all_n = 1 ;8-connectivity
     ;; count FG-objects
     nr_FG = max(label_region(ext1 eq 2b, all_neighbors=all_n, / ulong))     
     ;; extend image data assignment into the surrounding buffer to not create artificial boundaries at the image outline
     ext1[2:sz[0]+1, 1] = ext1[2:sz[0]+1, 2] ;; bottom row
     ext1[2:sz[0]+1, sz2[1]-2] = ext1[2:sz[0]+1, sz2[1]-3] ;; top row
     ext1[1, 2:sz[1]+1] = ext1[2, 2:sz[1]+1] ;; left column
     ext1[sz2[0]-2, 2:sz[1]+1] = ext1[sz2[0]-3, 2:sz[1]+1] ;; right column
     ;; assign the corner pixels
     ext1[1,1] = ext1[2,2]  ;; left bottom
     ext1[1,sz2[1]-2] = ext1[2,sz2[1]-3] ;;  left top
     ext1[sz2[0]-2,1] = ext1[sz2[0]-3,2] ;; right bottom 
     ext1[sz2[0]-2, sz2[1]-2] = ext1[sz2[0]-3, sz2[1]-3] ;; right top
     ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;     
     ;; FG
     fg = temporary(ext1) eq 2b ;; real FG
     ;; extend FG into missing to avoid getting boundaries there
     if image0min eq 0b then begin
      tmp = fg[2:sz[0]+1, 2:sz[1]+1]     
      fg[2:sz[0]+1, 2:sz[1]+1] = dilate(tmp, se8) * missing + temporary(tmp)
      missing = 0
     endif
     
     ;; spa2: small, linear features
     if eventValue2 eq 'spa2' then begin 
       slf = fg - morph_open(fg, se8)
       image0 = (fg * 17b) - (slf * 16b)
       image0 = temporary(image0[2:sz2(0)-3, 2:sz2(1)-3])
       if ct_qmiss gt 0 then image0[qmiss] = 129b 
       qmiss = 0
       zz = image0 eq 17b & nr_Cores = max(label_region(zz, all_neighbors=all_n, / ulong))  & zz = 0b
       zz = image0 eq 1b & nr_margin = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b
       goto, spa2skip0
     endif
     
     ;; Cores
     cores = erode(fg,se8)      
     
     ;; Margin    
     margin = fg - cores
     if eventValue2 eq 'spa3' then fg = 0b
     
     ;; Holes, use 8-conn for FG <-> 4-conn for BG
     ;; holes must be done on extended BG to ensure the actual background is a hole 
     holes = label_region(1b-cores,/ulong) 
     ;; holes=0 -> FG; holes=1 -> image BG; holes gt 1 = actual holes inside Cores
     qholes = where(holes gt 1, ct_qholes, /l64) & holes = 0
        
     ;; map for display     
     if eventValue2 eq 'spa3' then begin       
       image0 = temporary(cores)*17b
       if ct_qholes gt 0 then image0[qholes] = 100b & qholes = 0
       ;; overplot the margin pixels on top of the holes of Core      
       qmargin = where(margin eq 1b, /l64)
       image0[qmargin] = 1b & margin = 0 & qmargin = 0
     endif else begin 
       ct_qislets = 0
       
       ;; SPA6
       ;; at this point we have 3 data sets of extended size sz2:
       ;; - fg: the potential fixed fg at the intersection with missing
       ;; - cores: the core mask
       ;; - margin: all non-core FG
             
       if eventValue2 eq 'spa6' then begin
         ;; get the islets
         ;;=================        
          ;; reconstruction by dilation          
          cord = cores & sum1 = 1 & sum2 = 0 & steps = 0
          while sum2 ne sum1 do begin
            sum1 = sum2
            cord = dilate(cord, se8) * fg
            sum2 = total(cord)      
            steps = steps + 1     
          endwhile
          cord = fg - temporary(cord) 
          qislets = where(cord eq 1b, ct_qislets, /l64)
          cord = 0             
       endif
       
       ;; continue with spa5            
       core_boundary = dilate(cores,se8)*temporary(fg)-cores
       ;; start with Core
       image0 = temporary(cores)*17b
       ;; add core-holes
       if ct_qholes gt 0 then image0[qholes] = 100b & qholes = 0
       ;; add core_boundary
       margin = temporary(margin)-core_boundary & qmargin = where(margin gt 0, ct_qmargin, /l64) & margin = 0
       image0 = temporary(image0) + temporary(core_boundary)*3b      
       ;; assign Perforation
       q = where(image0 eq 103b, ct, /l64)
       if ct gt 0 then image0[q] = 5b
       q = 0
       ;; add margin pixels
       if ct_qmargin gt 0 then image0[qmargin] = 1b
       qmargin = 0   
       ;; add islet pixels
       if ct_qislets gt 0 then image0[qislets] = 9b
       qislets = 0    
     endelse
        
     ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 
     ;; only now after all classes are assigned do the statistics for holes in the extended image
     ;; do them with 4-connectivity because FG is fixed with 8-connectivity!
     ;; first, set the two box boundary to background, then do the labeling
     image0[0:1,*] = 0b & image0[sz2[0]-2:*,*] = 0b ;; left & right
     image0[*,0:1] = 0b & image0[*,sz2[1]-2:*] = 0b ;; bottom & top
     if ct_qmiss gt 0 then begin ;; add back the missing pixels
       zz = image0[2:sz2(0)-3, 2:sz2(1)-3]
       zz[qmiss] = 129b & qmiss = 0
       image0[2:sz2(0)-3, 2:sz2(1)-3] = temporary(zz)
     endif
     zz = image0 eq 100b & nr_holes = max(label_region(zz, / ulong)) & zz = 0b;; core-holes with 4-connectivity    
     zz = image0 eq 17b & nr_Cores = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b
     if eventValue2 eq 'spa3' then begin
       zz = image0 eq 1b & nr_margin = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b
     endif else begin ;; spa5/6
       ;; add the remaining stats: edge, perf, margin
       zz = image0 eq 3b & nr_edge = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b
       zz = image0 eq 5b & nr_perf = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b
       zz = image0 eq 1b & nr_margin = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b
       if eventValue2 eq 'spa6' then begin
         zz = image0 eq 9b & nr_islet = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b
       endif
       zz = image0 eq 17b or image0 eq 3b or image0 eq 5b & nr_contiguous = max(label_region(zz, all_neighbors=all_n, / ulong)) & zz = 0b    
     endelse

     ;; go back to original dimension 
     ;;=======================================
     image0 = temporary(image0[2:sz2(0)-3, 2:sz2(1)-3])
          
     spa2skip0:
     ;; get areas via histogram counting
     histo = histogram(image0,/l64)    
     area_miss = histo[129]
     area_Cores = histo[17]
     area_holes = histo[100]
     area_oBG = histo[0]
     area_bg = area_oBG + area_holes
     area_fg = area_data - area_bg

     ;; add the remaining stats: edge, perf, margin
     area_edge = histo[3]     
     area_perf = histo[5]     
     area_islet = histo[9]      
     area_margin = histo[1]
     area_contiguous = area_Cores + area_edge + area_perf
     area_internal = area_contiguous + area_holes
     porosity = 100.0 - (double(area_contiguous) / area_internal * 100.0)
             
     ;; write out statistics to disk in tmp to be copied later if image is saved
  openw, u, outdir + 'spa.raw', /get_lun & writeu, u, rotate(image0, 7) & free_lun, u
  openw, u, outdir + 'stats.txt', /get_lun
  printf, u, 'nr_FG=' + strtrim(nr_FG, 2)
  printf, u, 'nr_Cores=' + strtrim(nr_Cores, 2)
  printf, u, 'nr_margin=' + strtrim(nr_margin, 2)
  if n_elements(nr_holes) gt 0 then printf, u, 'nr_holes=' + strtrim(nr_holes, 2)
  if n_elements(nr_edge) gt 0 then printf, u, 'nr_edge=' + strtrim(nr_edge, 2)
  if n_elements(nr_perf) gt 0 then printf, u, 'nr_perf=' + strtrim(nr_perf, 2)
  if n_elements(nr_islet) gt 0 then printf, u, 'nr_islet=' + strtrim(nr_islet, 2)
  if n_elements(nr_contiguous) gt 0 then printf, u, 'nr_contiguous=' + strtrim(nr_contiguous, 2)
  printf, u, 'porosity=' + string(porosity, format='(G0.17)')
  free_lun, u
end
