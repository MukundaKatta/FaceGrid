"""Core FaceGrid class — face detection grid analyzer."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from pydantic import BaseModel

from facegrid.config import FaceGridConfig
from facegrid.utils import (
    aspect_ratio,
    bounding_box,
    compute_grid_skin_ratios,
    flood_fill_regions,
    grid_cell_to_pixel_box,
    rgb_to_hsv,
    skin_mask_from_hsv,
)


class Detection(BaseModel):
    """A single face-like region detection result."""

    region_id: int
    pixel_box: Tuple[int, int, int, int]  # (x0, y0, x1, y1)
    grid_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    area_cells: int
    aspect_ratio: float
    skin_density: float
    compactness: float
    score: float


class FaceGrid:
    """Face detection grid analyzer.

    Detects face-like regions in images by:
    1. Converting to HSV color space
    2. Creating a skin color mask
    3. Overlaying a grid and computing skin ratios per cell
    4. Finding connected regions of skin-heavy cells
    5. Scoring each region based on shape, density, and compactness

    No ML models are used — detection is purely heuristic.
    """

    def __init__(self, config: Optional[FaceGridConfig] = None) -> None:
        """Initialize FaceGrid with optional configuration.

        Args:
            config: Detection parameters. Uses defaults if not provided.
        """
        self.config = config or FaceGridConfig()

    def load_image(self, path: Union[str, Path]) -> Image.Image:
        """Load an image from disk and convert to RGB.

        Args:
            path: Path to the image file.

        Returns:
            PIL Image in RGB mode.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the file cannot be opened as an image.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as exc:
            raise ValueError(f"Cannot open image: {path}") from exc

    def detect_skin_regions(self, image: Image.Image) -> NDArray[np.bool_]:
        """Detect skin-colored pixels in an image.

        Converts the image to HSV space and applies dual-range thresholding
        to handle the hue wrap-around typical for human skin tones.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            Boolean numpy array of shape (H, W) marking skin pixels.
        """
        rgb_array = np.array(image, dtype=np.uint8)
        hsv = rgb_to_hsv(rgb_array)
        mask = skin_mask_from_hsv(
            hsv,
            self.config.hsv_lower,
            self.config.hsv_upper,
            self.config.hsv_lower_alt,
            self.config.hsv_upper_alt,
        )
        return mask

    def find_candidate_regions(
        self, skin_mask: NDArray[np.bool_]
    ) -> List[List[Tuple[int, int]]]:
        """Identify candidate face regions from a skin mask.

        Divides the skin mask into a grid, flags cells exceeding the
        skin ratio threshold, then groups contiguous flagged cells
        into candidate regions.

        Args:
            skin_mask: Boolean array of shape (H, W).

        Returns:
            List of regions, each a list of (row, col) grid coordinates.
        """
        ratios = compute_grid_skin_ratios(skin_mask, self.config.grid_size)
        flagged = ratios >= self.config.min_skin_ratio
        regions = flood_fill_regions(flagged)
        # Filter out regions smaller than min_region_area
        regions = [r for r in regions if len(r) >= self.config.min_region_area]
        return regions

    def analyze_region(
        self, region: List[Tuple[int, int]], skin_mask: NDArray[np.bool_]
    ) -> Dict[str, Any]:
        """Analyze a candidate region for geometric and density properties.

        Args:
            region: List of (row, col) grid cell coordinates.
            skin_mask: Boolean skin mask for computing density.

        Returns:
            Dictionary with keys: grid_box, area_cells, aspect_ratio,
            skin_density, compactness.
        """
        bbox = bounding_box(region)
        min_row, min_col, max_row, max_col = bbox
        ar = aspect_ratio(min_row, min_col, max_row, max_col)
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        compactness = len(region) / bbox_area if bbox_area > 0 else 0.0

        # Compute average skin density across the region cells
        h, w = skin_mask.shape
        cell_h = h / self.config.grid_size
        cell_w = w / self.config.grid_size
        densities = []
        for r, c in region:
            y0 = int(round(r * cell_h))
            y1 = int(round((r + 1) * cell_h))
            x0 = int(round(c * cell_w))
            x1 = int(round((c + 1) * cell_w))
            cell = skin_mask[y0:y1, x0:x1]
            if cell.size > 0:
                densities.append(float(np.mean(cell)))
        avg_density = float(np.mean(densities)) if densities else 0.0

        return {
            "grid_box": bbox,
            "area_cells": len(region),
            "aspect_ratio": ar,
            "skin_density": avg_density,
            "compactness": compactness,
        }

    def score_face_likelihood(self, analysis: Dict[str, Any]) -> float:
        """Score how likely a candidate region is to contain a face.

        Combines heuristics:
        - Aspect ratio close to 0.75 (typical face w/h ratio) scores higher
        - Higher compactness is better (faces are roughly elliptical)
        - Higher average skin density is better

        Args:
            analysis: Output from analyze_region().

        Returns:
            Float score in [0, 1]. Higher means more face-like.
        """
        # Aspect ratio score: ideal face is ~0.75 w/h
        ar = analysis["aspect_ratio"]
        ar_score = max(0.0, 1.0 - abs(ar - 0.75) / 1.5)

        # Compactness score
        compact_score = min(1.0, analysis["compactness"])

        # Skin density score
        density_score = min(1.0, analysis["skin_density"])

        # Area score: penalize very small regions
        area = analysis["area_cells"]
        area_score = min(1.0, area / 12.0)

        # Weighted combination
        score = (
            0.25 * ar_score
            + 0.25 * compact_score
            + 0.35 * density_score
            + 0.15 * area_score
        )
        return round(min(1.0, max(0.0, score)), 4)

    def detect_faces(self, image: Image.Image) -> List[Detection]:
        """Run the full face detection pipeline on an image.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            List of Detection objects for regions scoring above threshold.
        """
        skin_mask = self.detect_skin_regions(image)
        regions = self.find_candidate_regions(skin_mask)
        width, height = image.size

        detections: List[Detection] = []
        for idx, region in enumerate(regions):
            analysis = self.analyze_region(region, skin_mask)
            score = self.score_face_likelihood(analysis)

            if score < self.config.score_threshold:
                continue

            min_row, min_col, max_row, max_col = analysis["grid_box"]
            pixel_box = grid_cell_to_pixel_box(
                min_row, min_col, max_row, max_col, width, height, self.config.grid_size
            )

            detection = Detection(
                region_id=idx,
                pixel_box=pixel_box,
                grid_box=analysis["grid_box"],
                area_cells=analysis["area_cells"],
                aspect_ratio=round(analysis["aspect_ratio"], 4),
                skin_density=round(analysis["skin_density"], 4),
                compactness=round(analysis["compactness"], 4),
                score=score,
            )
            detections.append(detection)

        # Sort by score descending
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def draw_grid(
        self, image: Image.Image, detections: List[Detection]
    ) -> Image.Image:
        """Draw detection bounding boxes onto a copy of the image.

        Args:
            image: Original PIL Image.
            detections: List of Detection objects.

        Returns:
            New PIL Image with bounding boxes drawn.
        """
        output = image.copy()
        draw = ImageDraw.Draw(output)
        color = self.config.border_color
        width = self.config.border_width

        for det in detections:
            x0, y0, x1, y1 = det.pixel_box
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
            label = f"#{det.region_id} ({det.score:.2f})"
            text_y = max(0, y0 - 12)
            draw.text((x0, text_y), label, fill=color)

        return output

    def get_stats(self, detections: List[Detection]) -> Dict[str, Any]:
        """Compute summary statistics for a set of detections.

        Args:
            detections: List of Detection objects.

        Returns:
            Dictionary with total_detections, avg_score, max_score,
            min_score, avg_area_cells, and avg_skin_density.
        """
        if not detections:
            return {
                "total_detections": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "avg_area_cells": 0.0,
                "avg_skin_density": 0.0,
            }

        scores = [d.score for d in detections]
        areas = [d.area_cells for d in detections]
        densities = [d.skin_density for d in detections]

        return {
            "total_detections": len(detections),
            "avg_score": round(float(np.mean(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
            "min_score": round(float(np.min(scores)), 4),
            "avg_area_cells": round(float(np.mean(areas)), 2),
            "avg_skin_density": round(float(np.mean(densities)), 4),
        }

    def export(
        self,
        detections: List[Detection],
        format: str = "json",
        path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Export detections to JSON or CSV format.

        Args:
            detections: List of Detection objects.
            format: Output format — "json" or "csv".
            path: Optional file path to write to. If None, returns the string.

        Returns:
            Formatted string of the export data.

        Raises:
            ValueError: If format is not "json" or "csv".
        """
        if format not in ("json", "csv"):
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")

        records = [d.model_dump() for d in detections]

        if format == "json":
            output = json.dumps(records, indent=2, default=str)
        else:
            buf = io.StringIO()
            if records:
                writer = csv.DictWriter(buf, fieldnames=records[0].keys())
                writer.writeheader()
                for rec in records:
                    flat = {
                        k: str(v) if isinstance(v, tuple) else v for k, v in rec.items()
                    }
                    writer.writerow(flat)
            output = buf.getvalue()

        if path is not None:
            Path(path).write_text(output, encoding="utf-8")

        return output
