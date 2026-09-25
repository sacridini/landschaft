; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Batch Process -> Fragmentation -> FOS" for the
; landschaft test-suite: the processing statements of the 'batch_fos' event of
; guidostoolbox.pro (lines 6230-6455) copied verbatim, except for the
; progress bar, log and barplot lines. They run GTB's spatcon (spatcon.pro) and
; grayspatcon programs from <root>/spatcon/.
; GDL ignores HISTOGRAM's /L64 keyword, so the patch-area histograms are
; converted with long64() to get IDL's 64-bit counts (their squares overflow
; 32-bit integers for patches over 46340 pixels).
;
;   gtb_fos, 'in.tif', 'root/', 'out/', 'FAD_6class', kdim, fgconn ('8'/'4'), grayt, pseudo_bin

pro gtb_fos, input, root, outdir, fosclass, kdim, fgconn_str, grayt, pseudo_bin
  info = {dir_tmp: root + 'work/', my_os: 'windows', resfloat: 0, dir_guidossub: root, $
    mspa_param1_id: fgconn_str eq '8'}
  kdim_str = strtrim(kdim, 2) & grayt = byte(grayt)
  TT = ['Binary', 'Grayscale'] & fosinp = TT[1b - pseudo_bin]
  if strmid(fosclass, 4) eq '5class' then fostype = 'FOS5' else fostype = 'FOS6'
  image0 = read_tiff(input) & sz = size(image0,/dim)

       IF pseudo_bin THEN BEGIN ;; binary input to run (a) spatcon, (c) GSC52, or (e) FAC 
         qmiss = where(image0 eq 0b,ctmiss, /l64, complement=ruarea) & q3b = where(image0 eq 3b, ct3b, /l64) & q4b = where(image0 eq 4b, ct4b, /l64)
         BGmask = where(image0 EQ 1b, /l64) & qFG = where(image0 eq 2b, /l64, fgarea)
         n_ruarea = n_elements(ruarea)
         ;; get average patch size and # of patches
         ext1 = lonarr(sz[0] + 2, sz[1] + 2)
         ext1[1:sz[0], 1:sz[1]] = long(image0 eq 2b)
         conn8 = fgconn_str eq '8'
         ;; label FG only
         ext1 = label_region(temporary(ext1), all_neighbors=conn8, / ulong)
;         if strmid(fostype,0,7) eq 'FOS-APP' then obj_area = histogram(ext1, reverse_indices = rev, /l64) else obj_area = histogram(ext1, / l64)
         obj_area = histogram(ext1, / l64)
         obj_last=max(ext1) & z80 = strtrim(obj_last,2) 
         farea = total(obj_area[1:*],/double) & fareaperc=100.0/n_ruarea*farea
         aps = farea / obj_last & z81 = strtrim(aps,2) & obj_area = 0 
         z20 = '# Patches: ' + z80 & z22 = 'APS: ' + z81
         ;; get pixel indices by patch
         ext1 = histogram(temporary(ext1), /l64) & ext1 = long64(ext1)  ;; GDL: /l64 is ignored
         ;; PCnum:= overall connectivity. Sum of [ (areas per component)^2 ]
         pcnum = total(ext1(1: * )^2, / double) & ECA = sqrt(pcnum)
         ECA_max = total(ext1(1: * ), / double) & COH = ECA/ECA_max*100.0
         COH_ru = ECA/n_ruarea*100.0 & ext1=0       
         ;; calculate FAD/FAC for the fixed observation scale
         IF ct4b GT 0 THEN image0[q4b] = 0b ;; specialBG - assign to missing        
       ENDIF ELSE BEGIN ;; grayscale input to run GSC1 for (b, FAD) and GSC52 for (c, FED), (d, FED)
         qmiss = where(image0 eq 255b,ctmiss, /l64, complement=ruarea) & q3b = where(image0 eq 103b, ct3b, /l64) & q4b = where(image0 eq 104b, ct4b, /l64)
         BGmask = where(image0 LT grayt, ctbg, /l64) & n_ruarea = n_elements(ruarea)
         ;; exit if we have no BG or no FG
         IF ctbg EQ 0 THEN BEGIN
           GOTO, skip_batch_fos  ;; invalid input
         ENDIF 
         tt = image0 GE grayt AND image0 LT 101b & qFG = where(tt eq 1b, /l64, fgarea)
         IF fgarea EQ 0 THEN BEGIN
           GOTO, skip_batch_fos  ;; invalid input           
         ENDIF
         ;; the 'tt' is now the grayscale made binary, which we use to make the patch statistics with
         ;; get average patch size and # of patches; we do these statistics with the binary image
         ext1 = lonarr(sz[0] + 2, sz[1] + 2)
         ext1[1:sz[0], 1:sz[1]] = long(temporary(tt))
         IF info.mspa_param1_id EQ 1b THEN conn8 = 1 ELSE conn8 = 0
         ;; label FG only
         ext1 = label_region(ext1, all_neighbors=conn8, / ulong)
         obj_area = histogram(ext1, /l64)
         obj_area = histogram(ext1, /l64)
         obj_last=max(ext1) 
         farea = total(obj_area[1:*]) & fareaperc=100.0/n_ruarea*farea
         aps = farea / obj_last & z81 = strtrim(aps,2) & obj_area = 0 & z80 = strtrim(obj_last,2)
         z20 = '# Patches: ' + z80 & z22 = 'APS: ' + z81
         ;; get pixel indices by patch
         ext1 = histogram(temporary(ext1), /l64) & ext1 = long64(ext1)  ;; GDL: /l64 is ignored
         ;; PCnum:= overall connectivity. Sum of [ (areas per component)^2 ]
         pcnum = total(ext1(1: * )^2, / double) & ECA = sqrt(pcnum)
         ECA_max = total(ext1(1: * ), / double) & COH = ECA/ECA_max*100.0
         COH_ru = ECA/n_ruarea*100.0 & ext1=0         
         ;; calculate FAD/FED/FAC for the fixed observation scale
         IF ct3b GT 0 THEN image0[q3b] = 0b ;; set special BG to zero
         IF ct4b GT 0 THEN image0[q4b] = 255b ;; specialBG - assign to missing
       ENDELSE 
       fad_av = -1.0 & fadru_av = -1.0 ;; inititalise fad_av measure for foreground and reporting unit
       proc_spatcon = strmid(fosclass,0,3) EQ 'FAC' OR (strmid(fosclass,0,3) EQ 'FAD' AND fosinp EQ 'Binary')
       
       IF proc_spatcon EQ 1 THEN BEGIN
         ;; ***************  run spatcon PF (a) or FAC (e)  ***********************
         IF strmid(fosclass,0,3) EQ 'FAC' then mtyp='fac' else mtyp='pf'
         spatcon, image0, kdim, mtyp, info.dir_tmp, info.my_os, info.resfloat, im
         ;; rescale to normalized byte range
         if info.resfloat eq 0 then begin
           ;; normally the conversion to byte range would be: im=(im-1b)/254.0 > 0.0
           ;; the potential max value from spatcon is 255b and *only* those pixels can have a remapped value of 100b
           ; we must prevent that the value 254b will get rounded to 100b so mask the 255b pixels
           q = where(im eq 255b, ct, /l64)
           im = (temporary(im) - 1b)/254.0 & im = 0.994999 < temporary(im) > 0.0
           im = byte(round(temporary(im) * 100.0))
           if ct gt 0 then im[q] = 100b
         endif else begin
           im = byte(round(im*100.0))
         endelse  
       ENDIF ELSE BEGIN
         GSC1 = (strmid(fosclass,0,3) EQ 'FAD' AND fosinp EQ 'Grayscale')
         ;; ***************  run GSC for (b), (c), or (d) ***********************
         IF GSC1 EQ 1 THEN BEGIN ;; run GSC 1 for case (b)
           pushd, info.dir_tmp
           close, 1 & openw,1, 'gscpars.txt'
           printf,1,'R ' + strtrim(sz[1],2)
           printf,1,'C ' + strtrim(sz[0],2)
           printf,1,'M 1'
           printf,1,'P 0'
           printf,1,'G 0'
           printf,1,'W ' + kdim_str
           printf,1,'F 1'
           printf,1,'B 6'
           printf,1,'A 1'
           printf,1,'X 5'
           printf,1,'Y 10'
           printf,1,'K 5'
           close,1
         ENDIF ELSE BEGIN ;; run GSC 52 for (c) and (d)
           pushd, info.dir_tmp
           close, 1 & openw,1, 'gscpars.txt'
           printf,1,'R ' + strtrim(sz[1],2)
           printf,1,'C ' + strtrim(sz[0],2)
           printf,1,'M 52'
           printf,1,'P 0'
           printf,1,'G 0'
           printf,1,'W ' + kdim_str
           printf,1,'F 1'
           printf,1,'B 1'
           printf,1,'A 1'
           printf,1,'X 5'
           printf,1,'Y 10'
           printf,1,'K 5'
           close,1
           IF fosinp EQ 'Binary' THEN BEGIN ;; case (c), set all BG to zero
             image0[BGmask] = 0b 
             image0[qFG] = 100b
             image0[qmiss] = 255b
             IF ct3b GT 0 THEN image0[q3b] = 0b ;; set special BG to zero
             IF ct4b GT 0 THEN image0[q4b] = 255b ;; non-fragmenting specialBG - assign to missing
           ENDIF
         ENDELSE
         
         openw, 1, 'gscinput' & writeu,1, image0 & close,1 & image0 = 0
         ;; setup GraySpatCon
         IF info.my_os EQ 'windows' THEN BEGIN
           spatcon=info.dir_guidossub +'spatcon\grayspatcon64.exe' & file_copy, spatcon, 'grayspatcon.exe', /overwrite
         ENDIF ELSE IF info.my_os EQ 'darwin' THEN BEGIN
           spatcon=info.dir_guidossub +'spatcon/grayspatcon_mac' & file_copy, spatcon, 'grayspatcon', /overwrite
         ENDIF ELSE IF info.my_os EQ 'apple' THEN BEGIN
           spatcon=info.dir_guidossub +'spatcon/grayspatconARM_mac' & file_copy, spatcon, 'grayspatcon', /overwrite
         ENDIF ELSE BEGIN
           spatcon=info.dir_guidossub +'spatcon/grayspatcon_lin64' & file_copy, spatcon, 'grayspatcon', /overwrite
         ENDELSE
         ;; run grayspatcon in tmp
         IF info.my_os EQ 'windows' THEN spawn, 'grayspatcon.exe', log, / hide ELSE spawn, './grayspatcon', log
         ;; get result
         ;; if we get a GraySpatCon error then the last entry will not be "Normal Finish"
         res = log[n_elements(log)-1] & res = strpos(strlowcase(res), 'normal finish') gt 0
         if res eq 0 then begin
           file_delete, 'gscinput', 'gscoutput', 'gscoutput.txt', 'gscpars.txt', /allow_nonexistent,/quiet
           popd
           goto, skip_batch_fos
         endif
         ;; read the image output
         im = bytarr(sz(0),sz(1))
         openr, 1, 'gscoutput' & readu,1, im & close,1
         file_delete, 'gscinput', 'gscoutput', 'gscoutput.txt', 'gscpars.txt', /allow_nonexistent,/quiet
         popd
       ENDELSE  
       
       ;; calculate pixel-based fad_av BEFORE doing APP so they are consistent with non-APP
       fad_av = mean(im[qFG]) & fadru_av = fad_av * fgarea / n_ruarea  
       
       ;; add specialBG (105b), specialBG-Nf (106b), Missing (102b), background (101b)
       if ct3b gt 0 then im[q3b] = 105b & q3b = 0
       if ct4b gt 0 then im[q4b] = 106b & q4b = 0
       if ctmiss gt 0 then im[qmiss] = 102b
       qmiss = 0
       im[BGmask] = 101b & BGmask = 0
       
       ;; build histogram
       hist = histogram(im[qFG],/l64) & hist = hist[0:100]
       if strlen(fosclass) eq 10 then begin
         method = strmid(fosclass,0,3) & rep = strmid(fosclass,4) 
       endif else begin
         method = strmid(fosclass,0,7) & rep = strmid(fosclass,8) 
       endelse                    
       bins = findgen(101)-0.5 & xtit = method & tit = 'Foreground pixel histogram (WS: ' + kdim_str + ')' 
       ;; plot as percentage by FG-area
       hist2 = float(hist)/n_elements(qfg)*100.0 & bmax = max(hist2) * 1.05
       
       IF rep EQ '5class' THEN BEGIN
           
       ENDIF ELSE IF rep EQ '6class' THEN BEGIN
       ENDIF

       qFG = 0 
       fx = info.dir_tmp + 'fos.png' & file_delete,fx,/allow_nonexistent,/quiet
        
       
       ;; the statistics, first initialize
       intact = -1 & interior = -1 & dominant = -1 & transitional = -1
       patchy = -1 & rare = -1 & separated = -1 & continuous = -1
       if strlen(fostype) eq 4 then begin
         ;; get the 6 fragmentation proportions 
         if (fosclass eq 'FAD_6class') OR (fosclass eq 'FED_6class') OR (fosclass eq 'FAC_6class') then begin
           zz = (im EQ 100b) & intact = total(zz)/fgarea*100.0
           zz = (im GE 90b) AND (im LT 100b) & interior = total(zz)/fgarea*100.0
         endif else begin
           zz = (im GE 90b) AND (im LE 100b) & interior = total(zz)/fgarea*100.0
         endelse     
         zz = (im GE 60b) AND (im LT 90b) & dominant = total(zz)/fgarea*100.0
         zz = (im GE 40b) AND (im LT 60b) & transitional = total(zz)/fgarea*100.0
         zz = (im GE 10b) AND (im LT 40b) & patchy = total(zz)/fgarea*100.0 & z = 0
         zz = (im LT 10b) & rare = total(zz)/fgarea*100.0 & zz = 0
       endif 
       
  skip_batch_fos:
  openw, u, outdir + 'fos.raw', /get_lun & writeu, u, im & free_lun, u
  openw, u, outdir + 'stats.txt', /get_lun
  printf, u, 'fgarea=' + strtrim(fgarea, 2)
  printf, u, 'n_ruarea=' + strtrim(n_ruarea, 2)
  printf, u, 'obj_last=' + strtrim(obj_last, 2)
  printf, u, 'aps=' + string(aps, format='(G0.17)')
  printf, u, 'ECA=' + string(ECA, format='(G0.17)')
  printf, u, 'COH=' + string(COH, format='(G0.17)')
  printf, u, 'COH_ru=' + string(COH_ru, format='(G0.17)')
  printf, u, 'fareaperc=' + string(fareaperc, format='(G0.17)')
  printf, u, 'fad_av=' + string(fad_av, format='(G0.9)')
  printf, u, 'fadru_av=' + string(fadru_av, format='(G0.9)')
  printf, u, 'classes=' + strjoin(string([rare, patchy, transitional, dominant, interior, intact], format='(G0.9)'), ',')
  printf, u, 'hist2=' + strjoin(string(hist2, format='(G0.9)'), ',')
  free_lun, u
end
