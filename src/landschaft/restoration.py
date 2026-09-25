# SPDX-License-Identifier: GPL-3.0-or-later
"""Restoration Status (RSS) and Restoration Planner / Cost Distance analysis.

This module reimplements GuidosToolbox (GTB) restoration tools:
- :func:`restoration_status`: Evaluates landscape restoration status and metrics:
  Equivalent Connected Area (ECA), Degree of Coherence (DOC / COH), Restoration
  Potential (REST_POT), Relative Area of Connected Habitat (RAC), Average Patch
  Size (APS), and Critical New Object Area (CNOA).
- :func:`cost_distance`: Anisotropic 8-neighbor cost distance transform across
  resistance friction surfaces.
- :func:`least_cost_path`: Two-ended least-cost corridor path between habitat markers.
- :func:`restoration_corridor`: Comprehensive restoration corridor impact assessment
  evaluating connectivity gain (Delta-ECA, Delta-DOC) and restoration efficiency.

References
----------
Vogt, P. & Riitters, K. (2017). GuidosToolbox: universal digital image object analysis.
    European Journal of Remote Sensing, 50(1), 352-361. doi:10.1080/22797254.2017.1330650
Saura, S. & Pascual-Hortal, L. (2007). A new habitat availability index to integrate
    connectivity in landscape conservation planning. Landscape and Urban Planning, 83(2-3), 91-103.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import _core
from .io import load_raster

__all__ = [
    "restoration_status",
    "cost_distance",
    "least_cost_path",
    "restoration_corridor",
    "RestorationStatusResult",
    "LeastCostPathResult",
    "RestorationCorridorResult",
]


@dataclass(frozen=True)
class RestorationStatusResult:
    """Results from GuidosToolbox Restoration Status (RSS) analysis.

    Attributes
    ----------
    data_area : int
        Valid pixels in the image (total pixels minus missing / nodata).
    nr_comp : int
        Number of 8-connected foreground habitat components.
    eca_max : int
        Total foreground habitat area in pixels (:math:`\\sum A_i`).
    larg_obj : int
        Area of the largest foreground habitat component in pixels.
    aps : float
        Average Patch Size (:math:`ECA_{max} / N`).
    cnoa : int
        Critical New Object Area (:math:`\\lfloor (2 b c^2)/(b^2 - c^2) + 1 \\rfloor`).
    eca : float
        Equivalent Connected Area (:math:`\\sqrt{\\sum A_i^2}`).
    doc : float
        Degree of Coherence (:math:`(ECA / ECA_{max}) \\times 100\\%`).
    rest_pot : float
        Restoration Potential (:math:`100\\% - DOC`).
    rac : float
        Relative Area of Connected Habitat (:math:`(ECA_{max} / A_{valid}) \\times 100\\%`).
    """

    data_area: int
    nr_comp: int
    eca_max: int
    larg_obj: int
    aps: float
    cnoa: int
    eca: float
    doc: float
    rest_pot: float
    rac: float

    def to_dict(self) -> dict[str, Any]:
        """Return the restoration status metrics as a dictionary."""
        return {
            "data_area": self.data_area,
            "nr_comp": self.nr_comp,
            "eca_max": self.eca_max,
            "larg_obj": self.larg_obj,
            "aps": self.aps,
            "cnoa": self.cnoa,
            "eca": self.eca,
            "doc": self.doc,
            "rest_pot": self.rest_pot,
            "rac": self.rac,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return metrics formatted matching the GTB batch_rss spreadsheet."""
        return pd.DataFrame(
            [
                {
                    "AREA": self.data_area,
                    "RAC[%]": round(self.rac, 2),
                    "NR_OBJ": self.nr_comp,
                    "LARG_OBJ": self.larg_obj,
                    "APS": round(self.aps, 2),
                    "CNOA": self.cnoa,
                    "ECA": round(self.eca, 2),
                    "COH[%]": round(self.doc, 2),
                    "REST_POT[%]": round(self.rest_pot, 2),
                }
            ]
        )

    def summary(self) -> str:
        """Return a formatted text summary of the restoration status."""
        lines = [
            "Restoration Status (RSS) Summary",
            "================================",
            f"Valid Data Area [pixels]: {self.data_area:,}",
            f"Habitat Area (ECA_max) : {self.eca_max:,} ({self.rac:.2f}% of data area)",
            f"Number of Habitat Objects: {self.nr_comp:,}",
            f"Largest Habitat Object  : {self.larg_obj:,} pixels",
            f"Average Patch Size (APS): {self.aps:.2f} pixels",
            f"Critical New Object Area: {self.cnoa:,} pixels",
            f"Equivalent Connected Area: {self.eca:.2f} pixels",
            f"Degree of Coherence (DOC): {self.doc:.2f}%",
            f"Restoration Potential   : {self.rest_pot:.2f}%",
        ]
        return "\n".join(lines)


def restoration_status(
    data: xr.DataArray | np.ndarray | str | Path,
) -> RestorationStatusResult:
    """Compute GuidosToolbox Restoration Status (RSS) on a habitat map.

    Parameters
    ----------
    data : xarray.DataArray, numpy.ndarray, str, or Path
        Input habitat raster where:
        - 0: NoData / missing
        - 1: Background (or resistance values 1..100)
        - 2: Foreground habitat

    Returns
    -------
    RestorationStatusResult
        Computed restoration metrics including ECA, DOC, REST_POT, RAC, APS, and CNOA.
    """
    if isinstance(data, (str, Path)):
        data = load_raster(data)

    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("Input data must be a 2-D raster")

    u8_arr = np.ascontiguousarray(arr, dtype=np.uint8)
    res_dict = _core.compute_restoration_status(u8_arr)

    return RestorationStatusResult(
        data_area=int(res_dict["data_area"]),
        nr_comp=int(res_dict["nr_comp"]),
        eca_max=int(res_dict["eca_max"]),
        larg_obj=int(res_dict["larg_obj"]),
        aps=float(res_dict["aps"]),
        cnoa=int(res_dict["cnoa"]),
        eca=float(res_dict["eca"]),
        doc=float(res_dict["doc"]),
        rest_pot=float(res_dict["rest_pot"]),
        rac=float(res_dict["rac"]),
    )


def cost_distance(
    resistance: xr.DataArray | np.ndarray | str | Path,
    source_mask: xr.DataArray | np.ndarray | str | Path,
) -> xr.DataArray | np.ndarray:
    """Compute 8-neighbor anisotropic cost distance from source pixels across resistance.

    Parameters
    ----------
    resistance : xarray.DataArray, numpy.ndarray, str, or Path
        Resistance / friction surface (1..100, where 0 represents impassable barrier / nodata).
    source_mask : xarray.DataArray, numpy.ndarray, str, or Path
        Binary mask (> 0 indicates source cells from which cost distance is measured).

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Array of accumulated cost distance values.
    """
    is_xr = isinstance(resistance, xr.DataArray)
    xr_template = resistance if is_xr else None

    if isinstance(resistance, (str, Path)):
        resistance = load_raster(resistance)
        is_xr = True
        xr_template = resistance

    if isinstance(source_mask, (str, Path)):
        source_mask = load_raster(source_mask)

    res_arr = np.ascontiguousarray(np.asarray(resistance), dtype=np.uint8)
    src_arr = np.ascontiguousarray((np.asarray(source_mask) > 0).astype(np.uint8))

    dist_arr = _core.compute_cost_distance(res_arr, src_arr)

    if is_xr and xr_template is not None:
        return xr.DataArray(
            dist_arr,
            coords=xr_template.coords,
            dims=xr_template.dims,
            attrs={"long_name": "Cost Distance", "units": "accumulated resistance"},
        )
    return dist_arr


@dataclass(frozen=True)
class LeastCostPathResult:
    """Results from two-ended least-cost path calculation.

    Attributes
    ----------
    combined_cost : xarray.DataArray or numpy.ndarray
        Accumulated two-way cost field (Cost_A + Cost_B).
    lcp_mask : xarray.DataArray or numpy.ndarray
        Binary mask (1 on the least cost corridor path, 0 elsewhere).
    path_length : int
        Total length of the path in pixels.
    rest_pixels : int
        Number of restore pixels along the path located in the background.
    expense : float
        Total accumulated resistance along restore pixels.
    avg_distance : float
        Average distance along restore pixels (if distance map was provided).
    """

    combined_cost: xr.DataArray | np.ndarray
    lcp_mask: xr.DataArray | np.ndarray
    path_length: int
    rest_pixels: int
    expense: float
    avg_distance: float


def least_cost_path(
    resistance: xr.DataArray | np.ndarray | str | Path,
    marker_a: xr.DataArray | np.ndarray | str | Path,
    marker_b: xr.DataArray | np.ndarray | str | Path,
    dist_map: xr.DataArray | np.ndarray | None = None,
) -> LeastCostPathResult:
    """Compute the two-ended least cost path between marker A and marker B.

    Parameters
    ----------
    resistance : xarray.DataArray, numpy.ndarray, str, or Path
        Resistance friction surface (1..100, 0 = impassable).
    marker_a : xarray.DataArray, numpy.ndarray, str, or Path
        Source component / marker A mask.
    marker_b : xarray.DataArray, numpy.ndarray, str, or Path
        Target component / marker B mask.
    dist_map : xarray.DataArray or numpy.ndarray, optional
        Morphological distance transform map used to evaluate average distance quality.

    Returns
    -------
    LeastCostPathResult
        Container with combined cost field, LCP binary mask, and summary metrics.
    """
    is_xr = isinstance(resistance, xr.DataArray)
    xr_template = resistance if is_xr else None

    if isinstance(resistance, (str, Path)):
        resistance = load_raster(resistance)
        is_xr = True
        xr_template = resistance

    res_arr = np.ascontiguousarray(np.asarray(resistance), dtype=np.uint8)
    ma_arr = np.ascontiguousarray((np.asarray(marker_a) > 0).astype(np.uint8))
    mb_arr = np.ascontiguousarray((np.asarray(marker_b) > 0).astype(np.uint8))

    d_arr = None
    if dist_map is not None:
        d_arr = np.ascontiguousarray(np.asarray(dist_map), dtype=np.float32)

    res_dict = _core.compute_least_cost_path(res_arr, ma_arr, mb_arr, d_arr)

    cost_arr = res_dict["combined_cost"]
    lcp_arr = res_dict["lcp_mask"]

    if is_xr and xr_template is not None:
        cost_out = xr.DataArray(
            cost_arr,
            coords=xr_template.coords,
            dims=xr_template.dims,
            attrs={"long_name": "Combined Cost Distance", "units": "cost"},
        )
        lcp_out = xr.DataArray(
            lcp_arr,
            coords=xr_template.coords,
            dims=xr_template.dims,
            attrs={"long_name": "Least Cost Path", "values": "0: other, 1: path"},
        )
    else:
        cost_out = cost_arr
        lcp_out = lcp_arr

    return LeastCostPathResult(
        combined_cost=cost_out,
        lcp_mask=lcp_out,
        path_length=int(res_dict["path_length"]),
        rest_pixels=int(res_dict["rest_pixels"]),
        expense=float(res_dict["expense"]),
        avg_distance=float(res_dict["avg_distance"]),
    )


@dataclass(frozen=True)
class RestorationCorridorResult:
    """Comprehensive restoration corridor impact assessment.

    Attributes
    ----------
    lcp_result : LeastCostPathResult
        Underlying least-cost path result.
    size_a : int
        Pixel area of source component A.
    size_b : int
        Pixel area of target component B.
    rest_pixels : int
        Number of restored pixels needed to establish the corridor.
    expense : float
        Accumulated resistance cost of restored pixels.
    efficiency : float
        Restoration efficiency (:math:`\\Delta ECA / Expense`).
    eca_orig : float
        Landscape ECA before restoration.
    eca_new : float
        Landscape ECA after establishing the corridor.
    delta_eca : float
        Net increase in ECA (:math:`ECA_{new} - ECA_{orig}`).
    doc_orig : float
        Landscape Degree of Coherence (%) before restoration.
    doc_new : float
        Landscape Degree of Coherence (%) after restoration.
    delta_doc : float
        Net increase in Degree of Coherence (% points).
    """

    lcp_result: LeastCostPathResult
    size_a: int
    size_b: int
    rest_pixels: int
    expense: float
    efficiency: float
    eca_orig: float
    eca_new: float
    delta_eca: float
    doc_orig: float
    doc_new: float
    delta_doc: float

    def to_dataframe(self) -> pd.DataFrame:
        """Format metrics matching the GTB Restoration Planner spreadsheet."""
        return pd.DataFrame(
            [
                {
                    "SIZE_A": self.size_a,
                    "SIZE_B": self.size_b,
                    "REST_PIX": self.rest_pixels,
                    "AVDIST_RP": round(self.lcp_result.avg_distance, 2),
                    "EXPENSE": round(self.expense, 2),
                    "EFFIC": round(self.efficiency, 4),
                    "ECA_ORIG": round(self.eca_orig, 2),
                    "ECA_NEW": round(self.eca_new, 2),
                    "DELTA_ECA": round(self.delta_eca, 2),
                    "COH_ORIG": round(self.doc_orig, 2),
                    "COH_NEW": round(self.doc_new, 2),
                    "DELTA_COH": round(self.delta_doc, 2),
                }
            ]
        )


def restoration_corridor(
    data: xr.DataArray | np.ndarray | str | Path,
    resistance: xr.DataArray | np.ndarray | str | Path,
    marker_a: xr.DataArray | np.ndarray,
    marker_b: xr.DataArray | np.ndarray,
    dist_map: xr.DataArray | np.ndarray | None = None,
) -> RestorationCorridorResult:
    """Evaluate connectivity gain and efficiency of a restoration corridor.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Input habitat map (0=missing, 1=bg, 2=fg).
    resistance : xarray.DataArray or numpy.ndarray
        Resistance map (1..100, 0=impassable).
    marker_a : array-like
        Binary mask of component A.
    marker_b : array-like
        Binary mask of component B.
    dist_map : array-like, optional
        Morphological distance transform map.

    Returns
    -------
    RestorationCorridorResult
        Complete evaluation including ECA gain and restoration efficiency.
    """
    orig_rss = restoration_status(data)
    lcp_res = least_cost_path(resistance, marker_a, marker_b, dist_map=dist_map)

    # Calculate restored habitat map
    arr = np.asarray(data)
    lcp_mask = np.asarray(lcp_res.lcp_mask) > 0
    restored_data = np.copy(arr)
    # Restore background cells along LCP to foreground (2)
    restored_data[(arr == 1) & lcp_mask] = 2

    new_rss = restoration_status(restored_data)

    delta_eca = new_rss.eca - orig_rss.eca
    delta_doc = new_rss.doc - orig_rss.doc
    effic = (delta_eca / lcp_res.expense) if lcp_res.expense > 0 else 0.0

    size_a = int(np.sum(np.asarray(marker_a) > 0))
    size_b = int(np.sum(np.asarray(marker_b) > 0))

    return RestorationCorridorResult(
        lcp_result=lcp_res,
        size_a=size_a,
        size_b=size_b,
        rest_pixels=lcp_res.rest_pixels,
        expense=lcp_res.expense,
        efficiency=effic,
        eca_orig=orig_rss.eca,
        eca_new=new_rss.eca,
        delta_eca=delta_eca,
        doc_orig=orig_rss.doc,
        doc_new=new_rss.doc,
        delta_doc=delta_doc,
    )
