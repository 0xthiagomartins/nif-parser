"""
RG document processor.
"""

from __future__ import annotations

from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import time

from ..alignment import detect_decorative_borders_rg
from ..llm import extract_rg_from_text
from ..llm.schemas import RGExtraction
from .base import DocumentProcessor

try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False



class RGProcessor(DocumentProcessor):
    """Processor for RG documents."""


    def _get_console(self):
        return Console() if _HAS_RICH else None

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

        timing_start = time.perf_counter()
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

        def format_side(side: str, side_results: Dict) -> str:
            if max_variations is not None:
                selected = self._select_best_variations(side_results, max_variations)
                side_results = dict(side_results)
                side_results["variations"] = selected

            lines = []
            lines.append(f"--- {side.upper()} ---")
            for method_name, variation_result in side_results.get("variations", {}).items():
                if not variation_result.get("success", False):
                    continue
                text = self._clean_text(variation_result.get("raw_text", ""))
                if not text:
                    continue
                lines.append(f"Method: {method_name}")
                lines.append(text)
                lines.append("")
            return "\n".join(lines).strip()

        toon_parts = []
        if "front" in results:
            toon_parts.append(format_side("front", results["front"]))
        if "back" in results:
            toon_parts.append(format_side("back", results["back"]))

        toon_text = "\n\n".join(part for part in toon_parts if part)
        toon_elapsed = time.perf_counter() - timing_start

        if debug:
            output_path = debug_toon_path or "rg_combined.toon"
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(toon_text)

        llm_start = time.perf_counter()
        result = extract_rg_from_text(toon_text, debug=debug)
        llm_elapsed = time.perf_counter() - llm_start

        if debug:
            console = self._get_console()
            if console:
                table = Table(title="RG Debug Timing")
                table.add_column("Etapa")
                table.add_column("Tempo (s)", justify="right")
                table.add_row("preprocess + ocr + toon", f"{toon_elapsed:.2f}")
                table.add_row("llm -> estruturado", f"{llm_elapsed:.2f}")
                console.print(table)
            else:
                print("\n[DEBUG] Tempo preprocess + ocr + toon: %.2fs" % toon_elapsed)
                print("[DEBUG] Tempo LLM -> estruturado: %.2fs" % llm_elapsed)

        return result

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
