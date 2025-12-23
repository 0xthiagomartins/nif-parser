"""
RG document processor.
"""

from __future__ import annotations

from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..alignment import detect_decorative_borders_rg
from ..llm import extract_rg_from_text
from ..llm.schemas import RGExtraction
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

        variations = self._get_common_preprocessing_variations(img_no_border)
        variations.update(self._get_specific_preprocessing_variations(img_no_border))
        return variations


    def process(
        self,
        front_path: Optional[str] = None,
        back_path: Optional[str] = None,
        use_parallel: bool = True,
        max_variations: int = 8,
        debug: bool = False,
        debug_toon_path: Optional[str] = None,
    ) -> RGExtraction:
        if not front_path and not back_path:
            raise ValueError("Provide front_path or back_path")

        results = {}

        if use_parallel and front_path and back_path:
            with ThreadPoolExecutor(max_workers=2) as executor:
                front_future = executor.submit(self._process_image, front_path, True)
                back_future = executor.submit(self._process_image, back_path, True)
                results["front"] = front_future.result()
                results["back"] = back_future.result()
        else:
            if front_path:
                results["front"] = self._process_image(front_path, use_parallel)
            if back_path:
                results["back"] = self._process_image(back_path, use_parallel)

        toon_parts = []
        toon_parts.append("OCR Results (RG) - Combined")
        if "front" in results:
            toon_parts.append("--- FRONT ---")
            toon_parts.append(
                self._results_to_toon(
                    results["front"],
                    front_path or "",
                    "front",
                    max_variations=max_variations,
                )
            )
        if "back" in results:
            toon_parts.append("--- BACK ---")
            toon_parts.append(
                self._results_to_toon(
                    results["back"],
                    back_path or "",
                    "back",
                    max_variations=max_variations,
                )
            )

        toon_text = "\n".join(toon_parts)

        if debug:
            output_path = debug_toon_path or "rg_combined.toon"
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(toon_text)

        return extract_rg_from_text(toon_text)

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
