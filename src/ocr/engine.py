"""
OCR engine for processing images with Tesseract.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple

import numpy as np
import pytesseract


class OCREngine:
    """Engine for OCR processing."""

    def extract_text(self, img: np.ndarray) -> str:
        """
        Extract text using Tesseract.

        Args:
            img: Preprocessed image (numpy array)

        Returns:
            Extracted text
        """
        config = r"--oem 3 --psm 6 -l por+eng"
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()

    def _process_single_variation(self, args: Tuple[str, np.ndarray]) -> Tuple[str, Dict[str, Any]]:
        """
        Process a single preprocessing variation.

        Args:
            args: (method_name, processed_img)

        Returns:
            (method_name, result_dict)
        """
        method_name, processed_img = args
        start_time = time.time()

        try:
            text = self.extract_text(processed_img)
            processing_time = time.time() - start_time
            return (
                method_name,
                {
                    "raw_text": text,
                    "text_length": len(text),
                    "processing_time": processing_time,
                    "success": True,
                },
            )
        except Exception as exc:
            processing_time = time.time() - start_time
            return (
                method_name,
                {
                    "error": str(exc),
                    "raw_text": "",
                    "text_length": 0,
                    "processing_time": processing_time,
                    "success": False,
                },
            )

    def process_variations(
        self,
        variations: Dict[str, np.ndarray],
        use_parallel: bool = True,
        max_workers: int | None = None,
    ) -> Dict[str, Any]:
        """
        Process multiple preprocessing variations.

        Args:
            variations: {method_name: processed_img}
            use_parallel: If True, run in parallel
            max_workers: Max worker threads

        Returns:
            Processing results and metrics
        """
        overall_start = time.time()

        results: Dict[str, Any] = {
            "variations": {},
            "total_processing_time": 0.0,
            "parallel": use_parallel,
            "num_variations": len(variations),
        }

        try:
            if use_parallel:
                print(f"Processing {len(variations)} variations in parallel...")
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_method = {
                        executor.submit(self._process_single_variation, (name, img)): name
                        for name, img in variations.items()
                    }
                    for future in as_completed(future_to_method):
                        method_name, result = future.result()
                        results["variations"][method_name] = result
            else:
                print(f"Processing {len(variations)} variations sequentially...")
                for method_name, processed_img in variations.items():
                    _, result = self._process_single_variation((method_name, processed_img))
                    results["variations"][method_name] = result

            overall_time = time.time() - overall_start
            results["total_processing_time"] = overall_time

            successful = [r for r in results["variations"].values() if r.get("success", False)]
            if successful:
                avg_time = sum(r["processing_time"] for r in successful) / len(successful)
                results["avg_processing_time_per_variation"] = avg_time
                results["total_successful"] = len(successful)
                results["total_failed"] = len(results["variations"]) - len(successful)

            return results
        except Exception as exc:
            results["error"] = str(exc)
            results["total_processing_time"] = time.time() - overall_start
            return results
