"""
Tenuto - Expressive Score-to-Performance AI Engine
"""

from src.model import build_model, TenutoTransformer, TenutoBiGRU
from src.features import extract_40d_features_from_score
from src.alignment import compute_alignment_targets
from src.render import render_expressive_midi

try:
    from src.model_jax import TenutoTransformerJAX
except ImportError:
    TenutoTransformerJAX = None

__version__ = "0.1.0"
