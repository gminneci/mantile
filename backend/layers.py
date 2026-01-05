"""
DEPRECATED: This module has been refactored into a package structure.
============================================================================

This file is kept for backward compatibility. Please use:
    from backend.layers import Layer, Phase, DataType, LayerMetrics
    from backend.layers import MLPLayer, GatedMLPLayer
    from backend.layers import AttentionLayer, GroupedQueryAttentionLayer
    from backend.layers import NormLayer

New structure:
    backend/layers/
        __init__.py       - Package exports
        base.py          - Layer, Phase, DataType, LayerMetrics
        mlp.py           - MLPLayer, GatedMLPLayer
        attention.py     - AttentionLayer, GroupedQueryAttentionLayer
        norm.py          - NormLayer
"""

# Re-export everything from the new package structure for backward compatibility
from backend.layers import (
    Layer,
    Phase,
    DataType,
    LayerMetrics,
    MLPLayer,
    GatedMLPLayer,
    AttentionLayer,
    GroupedQueryAttentionLayer,
    NormLayer,
)

__all__ = [
    "Layer",
    "Phase",
    "DataType",
    "LayerMetrics",
    "MLPLayer",
    "GatedMLPLayer",
    "AttentionLayer",
    "GroupedQueryAttentionLayer",
    "NormLayer",
]
