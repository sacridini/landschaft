; SPDX-License-Identifier: GPL-3.0-or-later
;
; Reference run of GuidosToolbox "Change -> Landscape Mosaic Heatmap" for the
; landschaft test-suite: the statements of the 'delta_lm' event that build
; the heatmap difference and delta_heatmap.csv, copied verbatim from
; guidostoolbox.pro (lines 21011, 21478-21504). The heatmap.sav restores are
; replaced by reading hn.raw written by gtb_lm_heatmap.pro; the drawing is
; left out.
;
;   gtb_lm_delta, 'dir_a/', 'dir_b/', kdim_str_a, kdim_str_b, 'outdir/'

pro gtb_lm_delta, dir_a, dir_b, kdim_str_a, kdim_str_b, outdir
  info = {dir_tmp: outdir}
  hn_a = fltarr(256) & openr, u, dir_a + 'hn.raw', /get_lun & readu, u, hn_a & free_lun, u
  hn_b = fltarr(256) & openr, u, dir_b + 'hn.raw', /get_lun & readu, u, hn_b & free_lun, u

  hn = hn_b -hn_a  ;; the difference from A to B

       ;;==============================================
       hns = round(hn*1000)/1000.0 & dot = strpos(hns,'.') & hnss = strarr(n_elements(hns))
       for i = 0, n_elements(hns)-1 do hnss[i] = strmid(hns[i],0,dot[i]+4)
  
       openw,12, info.dir_tmp + 'delta_heatmap.csv'
       printf,12, 'Delta Landscape Mosaic: B(Window size: ' + kdim_str_b + ') - A(Window size: ' + kdim_str_a + ')'+ ', , , , , , , , , , , , , , , , , , , , ,'
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
  openw, u, outdir + 'hn.raw', /get_lun & writeu, u, float(hn) & free_lun, u
end
