; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "FAD-MS" (Fragmentation -> Multiple
; Observation Scale -> FAD-MS) for the landschaft test-suite.
;
; The processing statements are copied verbatim from the 'batch_fadms' event
; of guidostoolbox.pro; info.dir_tmp/my_os/resfloat became arguments, and the
; GeoTIFF/txt/csv/sav/png outputs are replaced by raw maps and a key=value
; text file. SPATCON (spatcon.pro) runs GTB's spatcon executable.
;
;   gtb_fad_mscale, 'in.tif', 'dir_tmp/', 'outdir/', 'windows'
;
; Outputs in outdir: fad_7.raw ... fad_243.raw, fad_mscale.raw (uint8, in the
; row order of the input file) and stats.txt.

pro gtb_fad_mscale, input, dir_tmp, outdir, my_os
  resfloat = 0
  image0 = read_tiff(input) & sz = size(image0,/dim)

  ;; image properties
  qmiss = where(image0 eq 0b, ctmiss, /l64, complement=ruarea) & n_ruarea = n_elements(ruarea)
  q3b = where(image0 eq 3b, ct3b, /l64) & q4b = where(image0 eq 4b, ct4b, /l64)
  BGmask = where(image0 EQ 1b, /l64) & FGmask = where(image0 eq 2b, /l64, fgarea)

  ;; loop over 5 observation scales
  kdim = [7, 13, 27, 81, 243]

  ;; define arrays for cummulative values in summary barplot in popup window
  intact = fltarr(6) & interior = intact & dominant = intact & transitional = intact
  patchy = intact & rare = intact & fad_av = intact & fadru_av = intact

  ;; calculate FAD for each of the 5 observation scales
  imdisp = image0 * 0 ;;; the sum of classes over observation scales image to be shown in the viewport

  for isc = 0,4 do begin
    if isc lt 4 then tmp = image0 else tmp = temporary(image0)
    IF ct4b GT 0 THEN tmp[q4b] = 0b
    kdim_str = strtrim(kdim[isc],2)

    ;; run spatcon PF
    spatcon, tmp, kdim[isc], 'pf', dir_tmp, my_os, resfloat, im
    ;; rescale to normalized byte range
    if resfloat eq 0 then begin
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
    fad_av(isc) = mean(im(FGmask))
    fadru_av[isc] = fad_av[isc]*fgarea/n_ruarea

    ;; add specialBG (105b), specialBG-Nf (106b), Missing (102b), background (101b)
    if ct3b gt 0 then im[q3b] = 105b
    if ct4b gt 0 then im[q4b] = 106b
    if ctmiss gt 0 then im[qmiss] = 102b
    im[BGmask] = 101b

    ;; the statistics
    ;; get the 5 fragmentation proportions (the 6th, rare, is always 100.0%)
    zz = (im EQ 100b) & intact(isc) = total(zz)/fgarea*100.0
    zz = (im GE 90b) AND (im LT 100b) & interior(isc) = total(zz)/fgarea*100.0
    zz = (im GE 60b) AND (im LT 90b) & dominant(isc) = total(zz)/fgarea*100.0
    zz = (im GE 40b) AND (im LT 60b) & transitional(isc) = total(zz)/fgarea*100.0
    zz = (im GE 10b) AND (im LT 40b) & patchy(isc) = total(zz)/fgarea*100.0
    zz = (im LT 10b) & rare(isc) = total(zz)/fgarea*100.0 & zz = 0

    ;; add to summary image
    imdisp = imdisp + im

    ;; write out the single scale image (replaces write_tiff)
    openw, u, outdir + 'fad_' + kdim_str + '.raw', /get_lun & writeu, u, im & free_lun, u
    im = 0
  endfor

  ;; average over the 5 classes in byte values
  im = byte(round(temporary(imdisp)/5.0))

  ;; add specialBG (105b), specialBG-Nf (106b), Missing (102b), background (101b) for display image: imdisp = actual summary data
  if ct3b gt 0 then im[q3b] = 105b
  if ct4b gt 0 then im[q4b] = 106b
  if ctmiss gt 0 then im[qmiss] = 102b
  im[BGmask] = 101b & BGmask = 0 & qmiss = 0 & q3b = 0 & q4b = 0
  fad_av[5] = mean(im[FGmask]) & FGmask = 0
  fadru_av[5] = fad_av[5]*fgarea/n_ruarea

  ;; stats for sum of classes over observation scale for display image: im
  zz = (im EQ 100b) & intact(5) = total(zz)/fgarea*100.0
  zz = (im GE 90b) AND (im LT 100b) & interior(5) = total(zz)/fgarea*100.0
  zz = (im GE 60b) AND (im LT 90b) & dominant(5) = total(zz)/fgarea*100.0
  zz = (im GE 40b) AND (im LT 60b) & transitional(5) = total(zz)/fgarea*100.0
  zz = (im GE 10b) AND (im LT 40b) & patchy(5) = total(zz)/fgarea*100.0
  zz = (im LT 10b) & rare(5) = total(zz)/fgarea*100.0 & zz = 0

  openw, u, outdir + 'fad_mscale.raw', /get_lun & writeu, u, im & free_lun, u

  ;; the numbers of the txt/csv/sav outputs
  openw, u, outdir + 'stats.txt', /get_lun
  printf, u, 'fgarea=' + strtrim(fgarea,2)
  printf, u, 'n_ruarea=' + strtrim(n_ruarea,2)
  printf, u, 'fad_av=' + strjoin(string(fad_av, format='(G0.9)'), ',')
  printf, u, 'fadru_av=' + strjoin(string(fadru_av, format='(G0.9)'), ',')
  printf, u, 'intact=' + strjoin(string(intact, format='(G0.9)'), ',')
  printf, u, 'interior=' + strjoin(string(interior, format='(G0.9)'), ',')
  printf, u, 'dominant=' + strjoin(string(dominant, format='(G0.9)'), ',')
  printf, u, 'transitional=' + strjoin(string(transitional, format='(G0.9)'), ',')
  printf, u, 'patchy=' + strjoin(string(patchy, format='(G0.9)'), ',')
  printf, u, 'rare=' + strjoin(string(rare, format='(G0.9)'), ',')
  free_lun, u
  ;; the float32 values themselves, for exact comparisons
  openw, u, outdir + 'fad_av.raw', /get_lun & writeu, u, fad_av, fadru_av & free_lun, u
end
