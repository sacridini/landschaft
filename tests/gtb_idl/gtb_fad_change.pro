; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Change -> FAD Multiscale" for the
; landschaft test-suite.
;
; The processing statements are copied verbatim from the 'backto_change_fad'
; part of the 'change' event of guidostoolbox.pro. Inputs are the outputs of
; gtb_fad_mscale.pro for times A and B (dir_a, dir_b) instead of the GTB
; *_fad_<scale>.tif / *_fad_mscale.sav files; the txt/csv/png reports are
; replaced by raw arrays:
;   change.raw  6 x (7 x 7) uint64  pixels A -> B (A class = row)
;   perc.raw    6 x (6 x 6) double  % of changed foreground pixels
;   sums.raw    6 x 8       double  sum_above, sum_below, farea_a, farea_b,
;                                   nc, forcom, diag, chpix
;   delta.raw   2 x 6       float   deltafad_av, deltafadru_av

pro gtb_fad_change, dir_a, dir_b, outdir
  kdim_str = ['', '7', '13', '27', '81', '243','mscale']
  fn_a = [''] & fn_b = ['']
  for isc = 1, 6 do begin
    fn_a = [fn_a, dir_a + 'fad_' + kdim_str[isc] + '.raw'] & fn_b = [fn_b, dir_b + 'fad_' + kdim_str[isc] + '.raw']
  endfor

  ;; replaces: restore, fn_a[7] / fn_b[7] (the fad_av and fadru_av of the sav files)
  av = fltarr(12) & openr, u, dir_a + 'fad_av.raw', /get_lun & readu, u, av & free_lun, u
  a_fad_av = av[0:5] & a_fadru_av = av[6:11]
  openr, u, dir_b + 'fad_av.raw', /get_lun & readu, u, av & free_lun, u
  b_fad_av = av[0:5] & b_fadru_av = av[6:11]
  fad_avok = 12 & fadru_avok = 12

  openr, u, outdir + 'size.txt', /get_lun & xdim = 0l & ydim = 0l & readf, u, xdim, ydim & free_lun, u
  all_change = ulon64arr(7, 7, 6) & all_perc = dblarr(6, 6, 6) & all_sums = dblarr(8, 6)

  for isc = 1, 6 do begin
    ;; loop over the 6 scales
    change = dblarr(7, 7) & ext = strtrim(kdim_str[isc],2)
    ;; replaces: im1 = read_tiff(fn_a[isc]) & im2 = read_tiff(fn_b[isc])
    im1 = bytarr(xdim, ydim) & openr, u, fn_a[isc], /get_lun & readu, u, im1 & free_lun, u
    im2 = bytarr(xdim, ydim) & openr, u, fn_b[isc], /get_lun & readu, u, im2 & free_lun, u
    farea_a = total(im1 LT 101b,/double) & farea_b = total(im2 LT 101b,/double) & nc = farea_b - farea_a
    forcom = total((im1 lE 100b)*(im2 LE 100b),/double)
    ;; potential values in scale images are the actual FAD values:
    ; 1-100, 101-BG, 102-missing, 105-special BG, 106-specialBG non-fragm.

    ;; exclude missing data from both maps, temporarily set them to 150b
    q = where(im1 eq 102b or im2 eq 102b, ctmiss, /l64)
    if ctmiss gt 0 then begin
      im1[q]=150b & im2[q]=150b
    endif
    ;; collapse all background types in im1 and im2
    q = where(im2 gt 100b and im2 lt 150b, ct, /l64) & if ct gt 0 then im2[q]=110b
    q = where(im1 gt 100b and im1 lt 150b, ct, /l64) & if ct gt 0 then im1[q]=110b


    ;; get pixels of each class type in im1 and look what they changed to in im2
    ch = dblarr(7) ; reset change vector and look for BG pixels in im1
    if ct gt 0 then begin ;; im1 has BG-pixels
      h=histogram(im2[q]) & ch = [h[110],total(h[0:9]),total(h[10:39]),total(h[40:59]),total(h[60:89]),total(h[90:99]),h[100]]
      change[*,0] = ch
    endif
    ch = ch*0 & q = where(im1 lt 10b, ct, /l64)
    if ct gt 0 then begin ;; im1 has rare pixels
      h=histogram(im2[q]) & ch = [h[110],total(h[0:9]),total(h[10:39]),total(h[40:59]),total(h[60:89]),total(h[90:99]),h[100]]
      change[*,1] = ch
    endif
    ch = ch*0 & q = where(im1 ge 10b and im1 le 39b, ct, /l64)
    if ct gt 0 then begin ;; im1 has patchy pixels
      h=histogram(im2[q]) & ch = [h[110],total(h[0:9]),total(h[10:39]),total(h[40:59]),total(h[60:89]),total(h[90:99]),h[100]]
      change[*,2] = ch
    endif
    ch = ch*0 & q = where(im1 ge 40b and im1 le 59b, ct, /l64)
    if ct gt 0 then begin ;; im1 has transitional pixels
      h=histogram(im2[q]) & ch = [h[110],total(h[0:9]),total(h[10:39]),total(h[40:59]),total(h[60:89]),total(h[90:99]),h[100]]
      change[*,3] = ch
    endif
    ch = ch*0 & q = where(im1 ge 60b and im1 le 89b, ct, /l64)
    if ct gt 0 then begin ;; im1 has dominant pixels
      h=histogram(im2[q]) & ch = [h[110],total(h[0:9]),total(h[10:39]),total(h[40:59]),total(h[60:89]),total(h[90:99]),h[100]]
      change[*,4] = ch
    endif
    ch = ch*0 & q = where(im1 ge 90b and im1 lt 100b, ct, /l64)
    if ct gt 0 then begin ;; im1 has interior pixels
      h=histogram(im2[q]) & ch = [h[110],total(h[0:9]),total(h[10:39]),total(h[40:59]),total(h[60:89]),total(h[90:99]),h[100]]
      change[*,5] = ch
    endif
    ch = ch*0 & q = where(im1 eq 100b, ct, /l64)
    if ct gt 0 then begin ;; im1 has intact pixels
      h=histogram(im2[q]) & ch = [h[110],total(h[0:9]),total(h[10:39]),total(h[40:59]),total(h[60:89]),total(h[90:99]),h[100]]
      change[*,6] = ch
    endif

    ;; calculate percentage of change in the sub-matrix of change
    mch = change[1:*,1:*]
    ;; null diagonal
    diag = 0
    for ix = 0,5 do begin
      for iy = 0,5 do begin
        if ix eq iy then begin
          diag = diag + mch[ix,iy]
          mch[ix,iy] = 0
        endif
      endfor
    endfor
    chpix = total(mch)
    perc = mch/chpix*100
    ;; sum of percentage with connectivity increase = above matrix diagonal
    sum_above = total(perc[0:*,0],/double) + total(perc[1:*,1],/double) + total(perc[2:*,2],/double) + total(perc[3:*,3],/double) + total(perc[4:*,4],/double)
    ;; sum of percentage with connectivity decrease = below matrix diagonal
    sum_below = total(perc,/double) - sum_above

    uchange = strtrim(ulong64(change),2) ; ;; get integer numbers
    deltafad_av = b_fad_av - a_fad_av
    deltafadru_av = b_fadru_av - a_fadru_av

    ;; replaces the txt/csv tables
    all_change[*, *, isc-1] = ulong64(change)
    all_perc[*, *, isc-1] = perc
    all_sums[*, isc-1] = [sum_above, sum_below, farea_a, farea_b, nc, forcom, diag, chpix]
  endfor ;; loop over observation scales

  openw, u, outdir + 'change.raw', /get_lun & writeu, u, all_change & free_lun, u
  openw, u, outdir + 'perc.raw', /get_lun & writeu, u, all_perc & free_lun, u
  openw, u, outdir + 'sums.raw', /get_lun & writeu, u, all_sums & free_lun, u
  openw, u, outdir + 'delta.raw', /get_lun & writeu, u, float(deltafad_av), float(deltafadru_av) & free_lun, u
end
