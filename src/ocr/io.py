"""
Image IO and display helpers.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image
from IPython.display import HTML, display

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image and convert to RGB.

    Args:
        image_path: Path to the image file

    Returns:
        RGB image as numpy array
    """
    import os

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def display_image(img: np.ndarray, max_width: int = 800) -> None:
    """
    Display image in notebook.

    Args:
        img: RGB image
        max_width: Max display width
    """
    height, width = img.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        img_display = cv2.resize(img, (new_width, new_height))
    else:
        img_display = img.copy()

    pil_img = Image.fromarray(img_display)
    display(pil_img)


def display_images_grid(images_dict: Dict[str, np.ndarray], cols: int = 3, max_width: int = 300) -> None:
    """
    Display multiple images in a grid.

    Args:
        images_dict: {name: image}
        cols: Number of columns
        max_width: Max width per image
    """
    if not images_dict:
        return

    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Showing images sequentially.")
        for name, img in images_dict.items():
            print(f"- {name.upper()}")
            display_image(img, max_width=max_width)
        return

    items = list(images_dict.items())
    num_images = len(items)
    rows = (num_images + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4, rows * 3))
    gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.3, wspace=0.3)

    for idx, (name, img) in enumerate(items):
        row = idx // cols
        col = idx % cols
        ax = fig.add_subplot(gs[row, col])

        height, width = img.shape[:2]
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            img_display = cv2.resize(img, (new_width, new_height))
        else:
            img_display = img.copy()

        if len(img_display.shape) == 2:
            ax.imshow(img_display, cmap="gray")
        else:
            ax.imshow(img_display)

        ax.set_title(name.upper(), fontsize=10, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def display_results_table(results: Dict[str, Dict[str, Any]], title: str = "Results") -> None:
    """
    Display results in an HTML table.
    """
    if not results:
        print("No results to display")
        return

    html = f"<h3>{title}</h3>"
    html += "<table border='1' cellpadding='8' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
    html += "<thead><tr style='background-color: #4CAF50; color: white;'>"
    html += "<th>Method</th><th>Status</th><th>Time (s)</th><th>Text (chars)</th>"
    html += "<th>Lines</th><th>Words</th>"
    html += "</tr></thead><tbody>"

    for method_name, result in results.items():
        success = result.get("success", False)
        status = "OK" if success else "Error"
        status_color = "#d4edda" if success else "#f8d7da"

        processing_time = result.get("processing_time", 0)
        text_length = result.get("text_length", 0)
        raw_text = result.get("raw_text", "")

        lines = len(raw_text.splitlines()) if raw_text else 0
        words = len(raw_text.split()) if raw_text else 0

        html += f"<tr style='background-color: {status_color};'>"
        html += f"<td><strong>{method_name.upper()}</strong></td>"
        html += f"<td>{status}</td>"
        html += f"<td>{processing_time:.2f}</td>"
        html += f"<td>{text_length}</td>"
        html += f"<td>{lines}</td>"
        html += f"<td>{words}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    display(HTML(html))


def display_results_table_with_images(
    results: Dict[str, Dict[str, Any]],
    images_dict: Dict[str, np.ndarray],
    title: str = "Results",
) -> None:
    """
    Display results with images in an HTML table.
    """
    if not results:
        print("No results to display")
        return

    methods = list(results.keys())
    if not methods:
        return

    html = f"<h3>{title}</h3>"
    html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"

    html += "<tr style='background-color: #f0f0f0;'>"
    html += "<th style='background-color: #4CAF50; color: white; padding: 10px;'>Image</th>"
    for method_name in methods:
        html += "<td style='text-align: center; padding: 10px; vertical-align: middle;'>"
        if method_name in images_dict:
            img = images_dict[method_name]
            height, width = img.shape[:2]
            max_table_width = 200
            if width > max_table_width:
                scale = max_table_width / width
                new_width = max_table_width
                new_height = int(height * scale)
                img_resized = cv2.resize(img, (new_width, new_height))
            else:
                img_resized = img.copy()

            if len(img_resized.shape) == 2:
                pil_img = Image.fromarray(img_resized, mode="L")
            else:
                pil_img = Image.fromarray(img_resized)

            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            html += f"<img src='data:image/png;base64,{img_str}' style='max-width: {max_table_width}px; height: auto;' />"
        html += "</td>"
    html += "</tr>"

    html += "<tr style='background-color: #e8f5e9;'>"
    html += "<th style='background-color: #4CAF50; color: white; padding: 10px;'>Method</th>"
    for method_name in methods:
        html += f"<td style='text-align: center; padding: 10px; font-weight: bold;'>{method_name.upper()}</td>"
    html += "</tr>"

    html += "<tr>"
    html += "<th style='background-color: #4CAF50; color: white; padding: 10px;'>Status</th>"
    for method_name in methods:
        result = results[method_name]
        success = result.get("success", False)
        status = "OK" if success else "Error"
        status_color = "#d4edda" if success else "#f8d7da"
        html += f"<td style='text-align: center; padding: 10px; background-color: {status_color};'>{status}</td>"
    html += "</tr>"

    html += "<tr>"
    html += "<th style='background-color: #4CAF50; color: white; padding: 10px;'>Time (s)</th>"
    for method_name in methods:
        result = results[method_name]
        processing_time = result.get("processing_time", 0)
        html += f"<td style='text-align: center; padding: 10px;'>{processing_time:.2f}</td>"
    html += "</tr>"

    html += "<tr>"
    html += "<th style='background-color: #4CAF50; color: white; padding: 10px;'>Text (chars)</th>"
    for method_name in methods:
        result = results[method_name]
        text_length = result.get("text_length", 0)
        html += f"<td style='text-align: center; padding: 10px;'>{text_length}</td>"
    html += "</tr>"

    html += "<tr>"
    html += "<th style='background-color: #4CAF50; color: white; padding: 10px;'>Lines</th>"
    for method_name in methods:
        result = results[method_name]
        raw_text = result.get("raw_text", "")
        lines = len(raw_text.splitlines()) if raw_text else 0
        html += f"<td style='text-align: center; padding: 10px;'>{lines}</td>"
    html += "</tr>"

    html += "<tr>"
    html += "<th style='background-color: #4CAF50; color: white; padding: 10px;'>Words</th>"
    for method_name in methods:
        result = results[method_name]
        raw_text = result.get("raw_text", "")
        words = len(raw_text.split()) if raw_text else 0
        html += f"<td style='text-align: center; padding: 10px;'>{words}</td>"
    html += "</tr>"

    html += "</table>"
    display(HTML(html))
