"""
Passaporte document processor.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from .base import DocumentProcessor


class PassaporteProcessor(DocumentProcessor):
    """Processor for Brazilian passports."""

    def _get_document_type(self) -> str:
        return "passaporte"

    def _preprocess_for_passaporte(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def _get_specific_preprocessing_variations(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        return {"passaporte_specific": self._preprocess_for_passaporte(img)}

    def extract_data(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        use_parallel: bool = True,
        max_variations: int = 8,
    ) -> str:
        print("Processing passaporte...")
        results = self._process_image(image_path, use_parallel=use_parallel)
        if output_path is None:
            output_path = "passaporte.toon"

        toon_content = self._results_to_toon(
            results, image_path, "single", max_variations=max_variations
        )

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(toon_content)

        print(f"Results saved at: {output_path}")
        return output_path
