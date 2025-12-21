"""
OCR for Brazilian documents (RG, CNH, Passaporte).
"""

from .ocr import (
    setup_tesseract,
    load_image,
    display_image,
    display_images_grid,
    display_results_table,
    display_results_table_with_images,
    detect_document_corners,
    correct_perspective,
    detect_rotation_angle,
    detect_rotation_angle_combined,
    rotate_image,
    auto_align_document,
    auto_correct_perspective_and_rotation,
    detect_decorative_borders_rg,
    OCREngine,
    DocumentProcessor,
    RGProcessor,
    CNHProcessor,
    PassaporteProcessor,
)

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
