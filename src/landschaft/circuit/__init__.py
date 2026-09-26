# SPDX-License-Identifier: GPL-3.0-or-later
"""Circuit-theory connectivity on rasters (Circuitscape, Omniscape)."""

from .advanced import AdvancedResult, advanced
from .omniscape import omniscape, source_from_resistance

__all__ = ["AdvancedResult", "advanced", "omniscape", "source_from_resistance"]
