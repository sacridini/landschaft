; SPDX-License-Identifier: GPL-3.0-or-later
;
; IDL's MORPH_OPEN, which GDL lacks: the documented opening, a dilation of the
; erosion by the same structuring element.
function morph_open, image, structure
  return, dilate(erode(image, structure), structure)
end
