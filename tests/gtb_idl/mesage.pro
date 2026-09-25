; SPDX-License-Identifier: GPL-3.0-or-later
;
; GDL's WRITE_TIFF calls MESAGE (a typo for MESSAGE) when COMPRESSION is set;
; GTB always sets it. This stub lets WRITE_TIFF go on (uncompressed).
pro mesage, text, informational=informational
end
