"""Tests for FaceGrid core functionality using synthetic images."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from facegrid import FaceGrid, FaceGridConfig


def _make_skin_tone_image(width: int = 128, height: int = 128) -> Image.Image:
    """Create a synthetic image with a skin-tone colored rectangle in the center."""
    img = Image.new("RGB", (width, height), color=(200, 200, 220))
    pixels = np.array(img)

    # Paint a skin-tone rectangle in the center (~40% of image)
    cy, cx = height // 2, width // 2
    rh, rw = height // 4, width // 5

    # Typical skin tone in RGB
    skin_rgb = (210, 160, 120)
    pixels[cy - rh : cy + rh, cx - rw : cx + rw] = skin_rgb

    return Image.fromarray(pixels)


def _make_uniform_blue_image(width: int = 128, height: int = 128) -> Image.Image:
    """Create a solid blue image with no skin tones."""
    return Image.new("RGB", (width, height), color=(30, 60, 200))


def _make_full_skin_image(width: int = 128, height: int = 128) -> Image.Image:
    """Create an image entirely filled with skin-like color."""
    img = np.full((height, width, 3), (200, 150, 110), dtype=np.uint8)
    return Image.fromarray(img)


class TestSkinDetection:
    """Tests for skin region detection."""

    def test_skin_mask_detects_skin_area(self) -> None:
        """Skin mask should have True values in the skin-colored region."""
        fg = FaceGrid()
        image = _make_skin_tone_image()
        mask = fg.detect_skin_regions(image)

        assert mask.shape == (128, 128)
        # The center should have skin pixels
        center_region = mask[32:96, 38:90]
        skin_ratio = np.mean(center_region)
        assert skin_ratio > 0.3, f"Expected skin in center, got ratio {skin_ratio}"

    def test_no_skin_in_blue_image(self) -> None:
        """A solid blue image should produce very few or no skin pixels."""
        fg = FaceGrid()
        image = _make_uniform_blue_image()
        mask = fg.detect_skin_regions(image)

        overall_ratio = np.mean(mask)
        assert overall_ratio < 0.05, f"Blue image should have no skin, got {overall_ratio}"


class TestDetection:
    """Tests for the full detection pipeline."""

    def test_detects_regions_in_skin_image(self) -> None:
        """Detection should find candidate regions in an image with skin-tone area."""
        config = FaceGridConfig(
            grid_size=8,
            min_skin_ratio=0.2,
            score_threshold=0.3,
            min_region_area=2,
        )
        fg = FaceGrid(config=config)
        image = _make_skin_tone_image(256, 256)
        detections = fg.detect_faces(image)

        assert len(detections) >= 1, "Should detect at least one region"
        for det in detections:
            assert det.score >= 0.3
            assert det.area_cells >= 2

    def test_no_detections_in_blue_image(self) -> None:
        """No face-like regions should be detected in a solid blue image."""
        fg = FaceGrid()
        image = _make_uniform_blue_image()
        detections = fg.detect_faces(image)

        assert len(detections) == 0, "Blue image should have no detections"

    def test_full_skin_image_detects_large_region(self) -> None:
        """An entirely skin-colored image should produce a detection with high density."""
        config = FaceGridConfig(
            grid_size=8,
            min_skin_ratio=0.2,
            score_threshold=0.3,
            min_region_area=2,
        )
        fg = FaceGrid(config=config)
        image = _make_full_skin_image(128, 128)
        detections = fg.detect_faces(image)

        assert len(detections) >= 1
        # The largest detection should cover many cells
        biggest = max(detections, key=lambda d: d.area_cells)
        assert biggest.area_cells >= 8
        assert biggest.skin_density > 0.5


class TestExportAndStats:
    """Tests for export and statistics."""

    def test_export_json(self) -> None:
        """Export to JSON should produce valid JSON."""
        config = FaceGridConfig(
            grid_size=8, min_skin_ratio=0.2, score_threshold=0.3, min_region_area=2
        )
        fg = FaceGrid(config=config)
        image = _make_skin_tone_image(256, 256)
        detections = fg.detect_faces(image)

        result = fg.export(detections, format="json")
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_export_csv(self) -> None:
        """Export to CSV should produce a non-empty string with headers."""
        config = FaceGridConfig(
            grid_size=8, min_skin_ratio=0.2, score_threshold=0.3, min_region_area=2
        )
        fg = FaceGrid(config=config)
        image = _make_full_skin_image(128, 128)
        detections = fg.detect_faces(image)

        result = fg.export(detections, format="csv")
        assert "region_id" in result
        assert "score" in result

    def test_stats_empty(self) -> None:
        """Stats on empty detections should return zero values."""
        fg = FaceGrid()
        stats = fg.get_stats([])
        assert stats["total_detections"] == 0
        assert stats["avg_score"] == 0.0

    def test_stats_with_detections(self) -> None:
        """Stats should summarize detection scores and areas."""
        config = FaceGridConfig(
            grid_size=8, min_skin_ratio=0.2, score_threshold=0.3, min_region_area=2
        )
        fg = FaceGrid(config=config)
        image = _make_full_skin_image(128, 128)
        detections = fg.detect_faces(image)

        stats = fg.get_stats(detections)
        assert stats["total_detections"] == len(detections)
        assert stats["avg_score"] > 0

    def test_draw_grid_returns_image(self) -> None:
        """draw_grid should return a PIL Image with same dimensions."""
        config = FaceGridConfig(
            grid_size=8, min_skin_ratio=0.2, score_threshold=0.3, min_region_area=2
        )
        fg = FaceGrid(config=config)
        image = _make_skin_tone_image(256, 256)
        detections = fg.detect_faces(image)

        result = fg.draw_grid(image, detections)
        assert isinstance(result, Image.Image)
        assert result.size == image.size

    def test_load_image(self) -> None:
        """load_image should load a saved image file."""
        fg = FaceGrid()
        img = _make_skin_tone_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            loaded = fg.load_image(f.name)
            assert loaded.size == img.size
            assert loaded.mode == "RGB"
            Path(f.name).unlink()

    def test_export_to_file(self) -> None:
        """Export should write to a file when path is given."""
        config = FaceGridConfig(
            grid_size=8, min_skin_ratio=0.2, score_threshold=0.3, min_region_area=2
        )
        fg = FaceGrid(config=config)
        image = _make_full_skin_image(128, 128)
        detections = fg.detect_faces(image)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fg.export(detections, format="json", path=f.name)
            content = Path(f.name).read_text()
            parsed = json.loads(content)
            assert isinstance(parsed, list)
            Path(f.name).unlink()
