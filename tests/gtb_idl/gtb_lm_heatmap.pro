; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Landscape Mosaic" heatmap for the landschaft
; test-suite: SPATCON rule 7 (spatcon.pro) on the input map, then the
; statements of GTB's HEATMAP procedure that build the 103-class histogram,
; the heatmap vector hn, hmap and heatmap.csv, copied verbatim from
; guidostoolbox.pro (lines 849-855, 892-893, 1250-1261, 1264-1294). The
; missing.sav restore is replaced by the qmiss variable and the recode and
; drawing steps are left out.
;
;   gtb_lm_heatmap, 'in.tif', 'dir_tmp', 'outdir/', kdim
;
; Outputs in outdir: im103.raw (uint8), hn.raw (float, 256), hmap.raw
; (float, 21 x 12) and heatmap.csv.

pro gtb_lm_heatmap, input, dir_tmp, outdir, kdim
  my_os = 'windows' & kdim_str = strtrim(kdim,2)
  image0 = read_tiff(input)
  ;; from the 'kernel_lm' event
  qmiss = where(image0 eq 0b,ctmiss, /l64)
  spatcon, image0, kdim, 'lm', dir_tmp, my_os, 0, im

  pushd, dir_tmp
  file_move, 'scsize.txt', 'recsize.txt', /overwrite
  close, 1 & openr,1, 'recsize.txt' & q=''
  readf,1,q & ydim = (strsplit(q,' ',/extract))[1]
  readf,1,q & xdim = (strsplit(q,' ',/extract))[1]
  close,1
  im103 = bytarr(xdim,ydim)
  openr, 1, 'scoutput' & readu,1, im103 & close,1
  ;; assign missing pixels if present before recoding
  if ctmiss gt 0 then im103[qmiss] = 0b  ;; replaces: restore, dir_tmp + 'missing.sav' & im103[qmiss] = 0b

  hist = histogram(im103) ;; the frequencies for each class
  tot = total(hist[1:236]) & hn = hist/tot*100.0 & eps = 0.000005 & hmax = max(hn[1:*])
  hmap = fltarr(21,12)-1.0
  hmap[10,0] = hn[170] & hmap[10,1] = hn[171]
  hmap[9:11,2] = [hn[91],hn[172],hn[81]]
  hmap[8:12,3] = [hn[93],hn[92],hn[121],hn[82],hn[83]]
  hmap[7:13,4] = [hn[95],hn[94],hn[124],hn[122],hn[123],hn[84],hn[85]]
  hmap[6:14,5] = [hn[152],hn[151],hn[217],hn[216],hn[215],hn[214],hn[213],hn[145],hn[144]]
  hmap[5:15,6] = [hn[154],hn[153],hn[219],hn[218],hn[233],hn[232],hn[231],hn[212],hn[211],hn[143],hn[142]]
  hmap[4:16,7] = [hn[65],hn[155],hn[221],hn[220],hn[235],hn[234],hn[236],hn[230],hn[229],hn[210],hn[209],hn[141],hn[55]]
  hmap[3:17,8] = [hn[63],hn[64],hn[113],hn[222],hn[223],hn[224],hn[225],hn[226],hn[227],hn[228],hn[207],hn[208],hn[104],hn[54],hn[53]]
  hmap[2:18,9] = [hn[61],hn[62],hn[111],hn[112],hn[114],hn[200],hn[201],hn[202],hn[203],hn[204],hn[205],hn[206],hn[103],hn[102],hn[101],hn[52],hn[51]]
  hmap[1:19,10] = [hn[191],hn[192],hn[71],hn[72],hn[73],hn[74],hn[75],hn[131],hn[132],hn[133],hn[134],hn[135],hn[45],hn[44],hn[43],hn[42],hn[41],hn[182],hn[181]]
  hmap[0,11] = [hn[190]] & hmap[20,11] = [hn[180]]

  ;; write csv output
  hns = round(hn*1000)/1000.0 & dot = strpos(hns,'.') & hnss = strarr(n_elements(hns))
  for i = 0, n_elements(hns)-1 do hnss[i] = strmid(hns[i],0,dot[i]+4)
  
  openw,12, 'heatmap.csv'
  if kdim_str eq 'multiscale' then begin
    printf,12, 'Landscape Mosaic (multiscale)' + ', , , , , , , , , , , , , , , , , , , , ,'
  endif else begin
    printf,12, 'Landscape Mosaic using Window size (' + kdim_str + 'x' + kdim_str +')' + ', , , , , , , , , , , , , , , , , , , , ,'
  endelse
  printf,12, ', , , , , , , , , , ,' +hnss[170]+ ', , , , , , , , , ,'
  printf,12, ', , , , , , , , , , ,'+hnss[171]+ ', , , , , , , , , ,'
  printf,12, ', , , , , , , , , ,'+hnss[91]+','+hnss[172]+','+hnss[81]+', , , , , , , , ,'
  printf,12, ', , , , , , , , ,'+hnss[93]+','+hnss[92]+','+hnss[121]+','+hnss[82]+','+hnss[83]+', , , , , , , ,'
  printf,12, ', , , , , , , ,'+hnss[95]+','+hnss[94]+','+hnss[124]+','+hnss[122]+','+hnss[123]+','+hnss[84]+','+hnss[85]+', , , , , , ,'
  printf,12, ', , , , , , ,'+hnss[152]+','+hnss[151]+','+hnss[217]+','+hnss[216]+','+hnss[215]+','+hnss[214]+','+$
    hnss[213]+','+hnss[145]+','+hnss[144]+', , , , , ,'
  printf,12, ', , , , , ,'+hnss[154]+','+hnss[153]+','+hnss[219]+','+hnss[218]+','+hnss[233]+','+hnss[232]+','+$
    hnss[231]+','+hnss[212]+','+hnss[211]+','+hnss[143]+','+hnss[142]+', , , , ,'
  printf,12, ', , , , ,'+hnss[65]+','+hnss[155]+','+hnss[221]+','+hnss[220]+','+hnss[235]+','+hnss[234]+','+hnss[236]+','+$
    hnss[230]+','+hnss[229]+','+hnss[210]+','+hnss[209]+','+hnss[141]+','+hnss[55]+ ', , , ,'
  printf,12, ', , , ,'+hnss[63]+','+hnss[64]+','+hnss[113]+','+hnss[222]+','+hnss[223]+','+hnss[224]+','+hnss[225]+','+$
    hnss[226]+','+hnss[227]+','+hnss[228]+','+hnss[207]+','+hnss[208]+','+hnss[104]+','+hnss[54]+','+hnss[53]+', , ,'
  printf,12, ', , ,'+hnss[61]+','+hnss[62]+','+hnss[111]+','+hnss[112]+','+hnss[114]+','+hnss[200]+','+hnss[201]+','+$
    hnss[202]+','+hnss[203]+','+hnss[204]+','+hnss[205]+','+hnss[206]+','+hnss[103]+','+hnss[102]+','+$
    hnss[101]+','+hnss[52]+','+hnss[51]+', ,'
  printf,12, ', ,'+hnss[191]+','+hnss[192]+','+hnss[71]+','+hnss[72]+','+hnss[73]+','+hnss[74]+','+hnss[75]+','+$
    hnss[131]+','+hnss[132]+','+hnss[133]+','+hnss[134]+','+hnss[135]+','+hnss[45]+','+hnss[44]+','+$
    hnss[43]+','+hnss[42]+','+hnss[41]+','+hnss[182]+','+hnss[181]+','
  printf,12, ','+hnss[190]+', , , , , , , , , , , , , , , , , , , ,'+hnss[180]
  close,12
  popd

  file_copy, dir_tmp + '/heatmap.csv', outdir + 'heatmap.csv', /overwrite
  openw, u, outdir + 'im103.raw', /get_lun & writeu, u, im103 & free_lun, u
  openw, u, outdir + 'hn.raw', /get_lun & writeu, u, float(hn) & free_lun, u
  openw, u, outdir + 'hmap.raw', /get_lun & writeu, u, float(hmap) & free_lun, u
end
