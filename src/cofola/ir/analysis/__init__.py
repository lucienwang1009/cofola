"""Analysis passes for the immutable IR.

This module provides analysis passes that compute derived information
from a Problem without mutating it:

- EntityAnalysis: Computes p_entities, max_size for all objects
- MaxSizeInference: Infers max_size from size constraints via LP
- BagClassification: Classifies bag entities as distinguishable/indistinguishable
"""

from .entities import EntityAnalysis, AnalysisResult, SetInfo, BagInfo
from .max_size import MaxSizeInference
from .bag_classify import BagClassification

__all__ = [
    "EntityAnalysis",
    "AnalysisResult",
    "SetInfo",
    "BagInfo",
    "MaxSizeInference",
    "BagClassification",
]