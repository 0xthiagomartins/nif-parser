"""
Document processors.
"""

from .base import DocumentProcessor
from .rg import RGProcessor
from .cnh import CNHProcessor
from .passaporte import PassaporteProcessor

__all__ = [
    "DocumentProcessor",
    "RGProcessor",
    "CNHProcessor",
    "PassaporteProcessor",
]
