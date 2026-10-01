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
from .io import _load_map
from ._save import Saveable

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
class RestorationStatusResult(Saveable):
    """Result of :func:`restoration_status`.

    Attributes
    ----------
    data_area : int
        Valid pixels (all but the missing ones).
    nr_comp : int
        Number of habitat objects (8-connected).
    eca_max : int
        Total habitat area, pixels.
    larg_obj : int
        Area of the largest object, pixels.
    aps : float
        Average patch size, eca_max / nr_comp.
    cnoa : int
        Critical new object area (GTB's formula), floor(2 b c² / (b² - c²) + 1)
        with b = eca_max and c = eca.
    eca : float
        Equivalent connected area, sqrt(sum of A_i²), pixels.
    doc : float
        Degree of coherence, ECA / eca_max in %.
    rest_pot : float
        Restoration potential, 100 - doc.
    rac : float
        Relative area of habitat, eca_max / data_area in %.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> status = ls.restoration_status(forest)
    >>> status.to_dataframe()[["NR_OBJ", "COH[%]", "REST_POT[%]"]]
       NR_OBJ  COH[%]  REST_POT[%]
    0    2850   51.64        48.36
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
        """The metrics as a dictionary.

        Returns
        -------
        dict
        """
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
        """The metrics as one row of GTB's ``batch_rss`` spreadsheet.

        Returns
        -------
        pandas.DataFrame
        """
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
        """Text summary of the restoration status.

        Returns
        -------
        str
        """
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
    """How connected the habitat is and how much reconnecting it could add (GTB's Restoration Status, ``batch_rss``).

    The habitat objects (8-connected foreground components) of areas A_i give:

    - ECA (equivalent connected area) = sqrt(sum of A_i²), the size of a single
      object with the same connectivity;
    - DOC (degree of coherence) = ECA / total habitat area, in %;
    - restoration potential = 100 - DOC;
    - RAC (relative area of habitat) = habitat area / valid area, in %;
    - APS (average patch size) and CNOA (critical new object area).

    Parameters
    ----------
    data
        Map coded 0 missing, 1 background (or resistance 1-100), 2 foreground
        (DataArray, array or path).

    Returns
    -------
    RestorationStatusResult

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> status = ls.restoration_status(forest)
    >>> status.nr_comp, round(status.doc, 1), round(status.rest_pot, 1)
    (2850, 51.6, 48.4)
    """
    if isinstance(data, (str, Path)):
        data = _load_map(data, validate="mspa")

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
    """Accumulated cost of travelling from the source cells across a resistance surface (8 neighbours).

    A step costs the mean resistance of the two cells times the step length (1,
    or sqrt(2) on the diagonals).

    Parameters
    ----------
    resistance
        Resistance surface (1-100; 0 = impassable / NoData): DataArray, array or
        path.
    source_mask
        Cells > 0 (or ``True``) are the sources, at cost 0.

    Returns
    -------
    xarray.DataArray or numpy.ndarray
        Accumulated cost; ``inf`` where the sources cannot be reached.

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> import numpy as np
    >>> resistance = ls.distance_to_resistance(forest)
    >>> objects = ls.accounting(forest)                     # the two largest forest objects
    >>> ids = objects.ids.values
    >>> marker = np.zeros(forest.shape, dtype="uint8")
    >>> marker[ids == objects.largest[0][0]] = 1            # object A
    >>> marker[ids == objects.largest[1][0]] = 2            # object B
    >>> marker[forest.values == 0] = 3                      # missing: not crossed
    >>> cost = ls.cost_distance(resistance, marker == 1)
    >>> round(float(cost.values[marker == 2].min()), 1)      # cheapest cost to reach B
    370.4
    """
    is_xr = isinstance(resistance, xr.DataArray)
    xr_template = resistance if is_xr else None

    if isinstance(resistance, (str, Path)):
        resistance = _load_map(resistance)
        is_xr = True
        xr_template = resistance

    if isinstance(source_mask, (str, Path)):
        source_mask = _load_map(source_mask)

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
class LeastCostPathResult(Saveable):
    """Result of :func:`least_cost_path`.

    Attributes
    ----------
    combined_cost : xarray.DataArray or numpy.ndarray
        Cost from A plus cost from B.
    lcp_mask : xarray.DataArray or numpy.ndarray
        1 on the path, 0 elsewhere.
    path_length : int
        Length of the path, pixels.
    rest_pixels : int
        Pixels of the path in the background: what would have to be restored.
    expense : float
        Total resistance of those pixels.
    avg_distance : float
        Mean distance of those pixels (with ``dist_map``; else 0).
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
    """The least-cost path between two areas A and B.

    The costs from A and from B (:func:`cost_distance`) are added; the path is
    the cells of minimal total cost.

    Parameters
    ----------
    resistance
        Resistance surface (1-100; 0 = impassable).
    marker_a, marker_b
        Masks of areas A and B.
    dist_map
        Optional distance map (e.g. :func:`euclidean_distance`) to report the
        mean distance of the restored cells.

    Returns
    -------
    LeastCostPathResult

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> import numpy as np
    >>> resistance = ls.distance_to_resistance(forest)
    >>> objects = ls.accounting(forest)                     # the two largest forest objects
    >>> ids = objects.ids.values
    >>> marker = np.zeros(forest.shape, dtype="uint8")
    >>> marker[ids == objects.largest[0][0]] = 1            # object A
    >>> marker[ids == objects.largest[1][0]] = 2            # object B
    >>> marker[forest.values == 0] = 3                      # missing: not crossed
    >>> path = ls.least_cost_path(resistance, marker == 1, marker == 2)
    >>> path.path_length, path.rest_pixels, path.expense
    (115, 113, 323.0)
    """
    is_xr = isinstance(resistance, xr.DataArray)
    xr_template = resistance if is_xr else None

    if isinstance(resistance, (str, Path)):
        resistance = _load_map(resistance)
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
class RestorationCorridorResult(Saveable):
    """Result of :func:`restoration_corridor`.

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
        Restoration efficiency, delta_eca / expense.
    eca_orig : float
        Landscape ECA before restoration.
    eca_new : float
        Landscape ECA after establishing the corridor.
    delta_eca : float
        Net increase in ECA, eca_new - eca_orig.
    doc_orig : float
        Landscape Degree of Coherence (%) before restoration.
    doc_new : float
        Landscape Degree of Coherence (%) after restoration.
    delta_doc : float
        Net increase in Degree of Coherence (% points).

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> import numpy as np
    >>> resistance = ls.distance_to_resistance(forest)
    >>> objects = ls.accounting(forest)                     # the two largest forest objects
    >>> ids = objects.ids.values
    >>> marker = np.zeros(forest.shape, dtype="uint8")
    >>> marker[ids == objects.largest[0][0]] = 1            # object A
    >>> marker[ids == objects.largest[1][0]] = 2            # object B
    >>> marker[forest.values == 0] = 3                      # missing: not crossed
    >>> corridor = ls.restoration_corridor(forest, resistance, marker == 1, marker == 2)
    >>> corridor.to_dataframe()[["REST_PIX", "COH_ORIG", "COH_NEW"]]
       REST_PIX  COH_ORIG  COH_NEW
    0       113     51.64    59.06
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
        """The metrics as one row of GTB's Restoration Planner spreadsheet.

        Returns
        -------
        pandas.DataFrame
        """
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
    """Connectivity gain of restoring the least-cost corridor between two areas (GTB's Restoration Planner).

    The corridor of :func:`least_cost_path` is turned into habitat and the
    landscape's ECA and degree of coherence (:func:`restoration_status`) are
    compared before and after.

    Parameters
    ----------
    data
        Habitat map coded 0 missing, 1 background, 2 foreground.
    resistance
        Resistance surface (1-100; 0 = impassable).
    marker_a, marker_b
        Masks of areas A and B.
    dist_map
        Optional distance map, as in :func:`least_cost_path`.

    Returns
    -------
    RestorationCorridorResult

    Examples
    --------
    >>> import landschaft as ls
    >>> forest = ls.load_raster(ls.example_data("input"))   # 0 missing, 1 background, 2 forest
    >>> import numpy as np
    >>> resistance = ls.distance_to_resistance(forest)
    >>> objects = ls.accounting(forest)                     # the two largest forest objects
    >>> ids = objects.ids.values
    >>> marker = np.zeros(forest.shape, dtype="uint8")
    >>> marker[ids == objects.largest[0][0]] = 1            # object A
    >>> marker[ids == objects.largest[1][0]] = 2            # object B
    >>> marker[forest.values == 0] = 3                      # missing: not crossed
    >>> corridor = ls.restoration_corridor(forest, resistance, marker == 1, marker == 2)
    >>> corridor.rest_pixels, round(corridor.delta_eca, 2), round(corridor.delta_doc, 2)
    (113, 31784.28, 7.41)
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
