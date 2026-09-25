; SPDX-License-Identifier: GPL-3.0-or-later
;
; GuidosToolbox's SPATCON procedure, copied verbatim from guidostoolbox.pro
; (lines 1302-1377). It runs the spatcon executable found at
; <dir_tmp>/../spatcon/spatcon64.exe (Windows), as in a GTB installation.

PRO spatcon, scinput, kdim, metric, dir_tmp, my_os, resfloat, scoutput
;; Spatial Convolution metrics by Kurt Riitters
;;
;; scinput: MSPA-compliant image
;; kdim: selected kernel size 
;; metric: selected metric (string)
;; dir_tmp: path to mspatmp
;; my_os: my_os
;; resfloat: output in float instead of default byte
;;
;; scoutput: output image

pushd, dir_tmp

case strlowCase(metric) of ;; Mapping rules, see spatcon 483ff
 'pf':  mstr='81' ;; FAD or Pf-Kurt
 'pff': mstr='77' ;; Pff-Kurt
 'fac': mstr = '76'  ;; FAC or foreground area clustering
 'lm': mstr='7' ;; landscape mosaic 103class
endcase


;; we need these 3 files in tmp-directory: scsize.txt, scpars.txt, scinput (bsq-file) -> scoutput
sz=size(scinput,/dim) & kstr = strtrim(kdim,2)
;; for spatcon: rows first, then columns!
openw,1, 'scsize.txt'
printf,1,'nrows '+strtrim(sz[1],2)
printf,1,'ncols '+strtrim(sz[0],2)
close,1

openw,1, 'scpars.txt'
printf,1,'w ' + kstr
printf,1,'r ' + mstr
printf,1,'a 2'
printf,1,'h 1'
if mstr eq '78' then begin
  printf,1,'b 3' 
endif else if mstr eq '76' then begin
  printf,1,'b 2' 
endif else begin
   printf,1,'b 0'
endelse
printf,1,'m 0'
if resfloat eq 1 then printf,1,'f 1' else printf,1,'f 0'
;;;PV if metric eq 'lm' then printf,1,'p 0'
close,1

openw, 1, 'scinput' & writeu,1, scinput & close,1

;; setup spatcon; note: we are in dir_tmp right now
IF my_os EQ 'windows' THEN BEGIN
  spatcon='..\spatcon\spatcon64.exe' & file_copy, spatcon, 'spatcon.exe', /overwrite
ENDIF ELSE IF my_os EQ 'darwin' THEN BEGIN
  spatcon='../spatcon/spatcon_mac' & file_copy, spatcon, 'spatcon', /overwrite
ENDIF ELSE IF my_os EQ 'apple' THEN BEGIN
  spatcon='../spatcon/spatconARM_mac' & file_copy, spatcon, 'spatcon', /overwrite
ENDIF ELSE BEGIN
  spatcon='../spatcon/spatcon_lin64' & file_copy, spatcon, 'spatcon', /overwrite
ENDELSE


;; run spatcon in tmp
IF my_os EQ 'windows' THEN spawn, 'spatcon.exe', log, / hide ELSE spawn, './spatcon', log

;; if NOT LM then get the result and clean up
;; when doing LM the intermediate files are processed further
if mstr ne '7' then begin
  ;; get result
  scoutput = bytarr(sz(0),sz(1)) & scinput=0
  if resfloat eq 1 then scoutput=float(scoutput)
  openr, 1, 'scoutput' & readu,1, scoutput & close,1
    
endif 
popd

END
