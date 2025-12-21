"""
CNH document processor.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .base import DocumentProcessor


class CNHProcessor(DocumentProcessor):
    """Processor for CNH documents."""

    def _get_document_type(self) -> str:
        return "cnh"

    def _preprocess_for_cnh(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def _get_specific_preprocessing_variations(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        return {"cnh_specific": self._preprocess_for_cnh(img)}

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
