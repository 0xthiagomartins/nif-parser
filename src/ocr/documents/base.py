"""
Base document processor with common OCR and preprocessing flow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..engine import OCREngine
from ..io import load_image


class DocumentProcessor(ABC):
    """Base class for document processing."""

    def __init__(self) -> None:
        self.ocr_engine = OCREngine()
        self.document_type = self._get_document_type()

    @abstractmethod
    def _get_document_type(self) -> str:
        """Return document type (rg, cnh, passaporte)."""

    def _preprocess_default(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _preprocess_adaptive(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def _preprocess_improved(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned

    def _preprocess_light(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _get_common_preprocessing_variations(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        variations: Dict[str, np.ndarray] = {}
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        variations["default"] = self._preprocess_default(img)
        variations["adaptive"] = self._preprocess_adaptive(img)
        variations["improved"] = self._preprocess_improved(img)
        variations["light"] = self._preprocess_light(img)

        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        variations["adaptive_large"] = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
        )
        variations["adaptive_small"] = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2
        )

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variations["otsu_clahe"] = binary

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variations["otsu_blur"] = binary

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        variations["improved_aggressive"] = cleaned

        variations["grayscale"] = gray
        return variations

    @abstractmethod
    def _get_specific_preprocessing_variations(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Return preprocessing variations specific to a document type."""

    def _preprocess_image(self, image_path: str) -> Dict[str, np.ndarray]:
        img = load_image(image_path)
        variations = self._get_common_preprocessing_variations(img)
        variations.update(self._get_specific_preprocessing_variations(img))
        return variations

    def _process_image(
        self,
        image_path: str,
        use_parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        variations = self._preprocess_image(image_path)
        results = self.ocr_engine.process_variations(
            variations=variations,
            use_parallel=use_parallel,
            max_workers=max_workers,
        )
        results["image_path"] = image_path
        results["document_type"] = self.document_type
        return results

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        cleaned_lines = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            cleaned_lines.append(line)
            prev_empty = is_empty

        return "\n".join(cleaned_lines)

    def _select_best_variations(self, results: Dict[str, Any], max_variations: int) -> Dict[str, Any]:
        valid_variations: Dict[str, Any] = {}
        for method_name, variation_result in results.get("variations", {}).items():
            if variation_result.get("success", False):
                text = self._clean_text(variation_result.get("raw_text", ""))
                if text:
                    valid_variations[method_name] = variation_result

        if len(valid_variations) <= max_variations:
            return valid_variations

        sorted_variations = sorted(
            valid_variations.items(),
            key=lambda item: item[1].get("text_length", 0),
            reverse=True,
        )
        return dict(sorted_variations[:max_variations])

    def _results_to_toon(
        self,
        results: Dict[str, Any],
        image_path: str,
        side: str,
        max_variations: Optional[int] = None,
    ) -> str:
        lines = []

        if max_variations is not None:
            selected_variations = self._select_best_variations(results, max_variations)
            filtered_results = results.copy()
            filtered_results["variations"] = selected_variations
            results = filtered_results

        num_ocr_performed = sum(
            1
            for v in results.get("variations", {}).values()
            if v.get("success", False) and self._clean_text(v.get("raw_text", ""))
        )

        lines.append(
            f"OCR Results: {self.document_type.upper()} {side} | Image: {image_path} | Methods: {num_ocr_performed}"
        )
        lines.append("")
        lines.append("Each method below shows OCR text extracted using different image preprocessing:")
        lines.append("")

        for method_name, variation_result in results.get("variations", {}).items():
            if variation_result.get("success", False):
                text = self._clean_text(variation_result.get("raw_text", ""))
                if text:
                    lines.append(f"Method: {method_name}")
                    lines.append("OCR Text:")
                    lines.append(text)
                    lines.append("")

        return "\n".join(lines)

    def front_to_toon(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        use_parallel: bool = True,
        max_variations: int = 8,
    ) -> str:
        if output_path is None:
            output_path = f"{self.document_type}_front.toon"

        results = self._process_image(image_path, use_parallel=use_parallel)
        toon_content = self._results_to_toon(results, image_path, "front", max_variations=max_variations)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(toon_content)

        print(f"Results saved at: {output_path}")
        return output_path

    def back_to_toon(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        use_parallel: bool = True,
        max_variations: int = 8,
    ) -> str:
        if output_path is None:
            output_path = f"{self.document_type}_back.toon"

        results = self._process_image(image_path, use_parallel=use_parallel)
        toon_content = self._results_to_toon(results, image_path, "back", max_variations=max_variations)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(toon_content)

        print(f"Results saved at: {output_path}")
        return output_path

    def to_toon_pair(
        self,
        front_path: str,
        back_path: str,
        output_path: Optional[str] = None,
        use_parallel: bool = True,
        max_variations: int = 8,
    ) -> str:
        print(f"Processing {self.document_type.upper()}: front and back in parallel...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            front_future = executor.submit(self._process_image, front_path, use_parallel)
            back_future = executor.submit(self._process_image, back_path, use_parallel)
            front_results = front_future.result()
            back_results = back_future.result()

        if output_path is None:
            front_output = f"{self.document_type}_front.toon"
            back_output = f"{self.document_type}_back.toon"
        else:
            base_path = output_path.replace(".toon", "")
            front_output = f"{base_path}_front.toon"
            back_output = f"{base_path}_back.toon"

        front_toon = self._results_to_toon(
            front_results, front_path, "front", max_variations=max_variations
        )
        with open(front_output, "w", encoding="utf-8") as file:
            file.write(front_toon)

        back_toon = self._results_to_toon(back_results, back_path, "back", max_variations=max_variations)
        with open(back_output, "w", encoding="utf-8") as file:
            file.write(back_toon)

        print(f"Results saved at: {front_output} and {back_output}")
        return front_output
