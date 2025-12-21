"""
OCR package for Brazilian documents.
"""

from .config import setup_tesseract
from .io import (
    load_image,
    display_image,
    display_images_grid,
    display_results_table,
    display_results_table_with_images,
)
from .alignment import (
    detect_document_corners,
    correct_perspective,
    detect_rotation_angle,
    detect_rotation_angle_combined,
    rotate_image,
    auto_align_document,
    auto_correct_perspective_and_rotation,
    detect_decorative_borders_rg,
)
from .engine import OCREngine
from .documents.base import DocumentProcessor
from .documents.rg import RGProcessor
from .documents.cnh import CNHProcessor
from .documents.passaporte import PassaporteProcessor

__all__ = [
    "setup_tesseract",
    "load_image",
    "display_image",
    "display_images_grid",
    "display_results_table",
    "display_results_table_with_images",
    "detect_document_corners",
    "correct_perspective",
    "detect_rotation_angle",
    "detect_rotation_angle_combined",
    "rotate_image",
    "auto_align_document",
    "auto_correct_perspective_and_rotation",
    "detect_decorative_borders_rg",
    "OCREngine",
    "DocumentProcessor",
    "RGProcessor",
    "CNHProcessor",
    "PassaporteProcessor",
]
