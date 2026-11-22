"""nichepeek: see what cell-type label error does to your spatial niche analysis.

    import nichepeek as npk
    res = npk.diagnose(adata, label_key="celltype", target="^Fib", section_key="sample",
                      spatial="spatial", counts_layer="counts", out_dir="nichepeek_out")
"""
from .api import Result, diagnose
from .config import Config
from .labeling import CentroidLabeler

__all__ = ["diagnose", "Result", "Config", "CentroidLabeler"]
__version__ = "0.3.5"
