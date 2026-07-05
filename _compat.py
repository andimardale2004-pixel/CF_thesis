"""NumPy 2.0 compatibility shim sham for prfpy's legacy code.

Import this BEFORE importing prfpy. It re-adds the aliases NumPy 2.0 removed
(np.Inf, np.int, ...). On numpy<2 these already exist, so it is a no-go.
"""
import numpy as np

_LEGACY = {"Inf": np.inf, "NaN": np.nan, "NAN": np.nan,
           "float": float, "int": int, "bool": bool, "object": object, "str": str}
for _name, _val in _LEGACY.items():
    if not hasattr(np, _name):
        setattr(np, _name, _val)
