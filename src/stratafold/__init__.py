"""StrataFold control plane.

StrataFold studies structural MoE compression without additional quantization.
The target checkpoint's native mixed-precision representation is always the
baseline; this package never describes it as unquantized.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.0.0"
