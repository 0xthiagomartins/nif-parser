"""
RG document processor.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..alignment import detect_decorative_borders_rg
from .base import DocumentProcessor


class RGProcessor(DocumentProcessor):
    """Processor for RG documents."""

    def _get_document_type(self) -> str:
        return "rg"

    def _get_specific_preprocessing_variations(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        return {}

    def _preprocess_image(self, image_path: str) -> Dict[str, np.ndarray]:
        from ..io import load_image

        img = load_image(image_path)
        original_shape = img.shape

        top_crop, bottom_crop, left_crop, right_crop = detect_decorative_borders_rg(img)

        if (
            top_crop < 0
            or bottom_crop > img.shape[0]
            or left_crop < 0
            or right_crop > img.shape[1]
        ):
            print("RG border detection returned invalid values, using original image.")
            img_no_border = img
        else:
            img_no_border = img[top_crop:bottom_crop, left_crop:right_crop]

        print("RG border removal debug:")
        print(f"  Original shape: {original_shape}")
        print(
            f"  Crop values: top={top_crop}, bottom={bottom_crop}, left={left_crop}, right={right_crop}"
        )
        print(f"  Cropped shape: {img_no_border.shape}")
        print(
            "  Size reduction: "
            f"{original_shape[0] * original_shape[1] - img_no_border.shape[0] * img_no_border.shape[1]} pixels"
        )

        variations = self._get_common_preprocessing_variations(img_no_border)
        variations.update(self._get_specific_preprocessing_variations(img_no_border))
        return variations

    def extract_data(
        self,
        front_path: str,
        back_path: str,
        output_path: str | None = None,
        use_parallel: bool = True,
        max_variations: int = 8,
    ) -> str:
        return self.to_toon_pair(
            front_path,
            back_path,
            output_path=output_path,
            use_parallel=use_parallel,
            max_variations=max_variations,
        )
