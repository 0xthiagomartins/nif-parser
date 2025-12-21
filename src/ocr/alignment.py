"""
Alignment helpers (re-exported from legacy module).
"""

from ..alignment import (
    detect_document_corners,
    correct_perspective,
    detect_rotation_angle,
    detect_rotation_angle_combined,
    rotate_image,
    auto_align_document,
    auto_correct_perspective_and_rotation,
    detect_decorative_borders_rg,
)

__all__ = [
    "detect_document_corners",
    "correct_perspective",
    "detect_rotation_angle",
    "detect_rotation_angle_combined",
    "rotate_image",
    "auto_align_document",
    "auto_correct_perspective_and_rotation",
    "detect_decorative_borders_rg",
]
