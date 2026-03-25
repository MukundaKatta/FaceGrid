# FaceGrid Architecture

## Overview

FaceGrid is a heuristic face detection library that uses skin color analysis in HSV color space combined with grid-based region segmentation. It requires no machine learning models.

## Pipeline

```
Input Image (RGB)
    |
    v
HSV Conversion (utils.rgb_to_hsv)
    |
    v
Skin Mask Generation (utils.skin_mask_from_hsv)
    |   - Dual HSV range thresholding
    |   - Handles hue wrap-around for skin tones
    v
Grid Overlay (utils.compute_grid_skin_ratios)
    |   - Divides image into NxN grid cells
    |   - Computes skin pixel ratio per cell
    v
Region Discovery (utils.flood_fill_regions)
    |   - Flags cells above min_skin_ratio
    |   - Flood-fill to find connected components
    v
Region Analysis (core.FaceGrid.analyze_region)
    |   - Bounding box, aspect ratio, compactness
    |   - Average skin density
    v
Face Likelihood Scoring (core.FaceGrid.score_face_likelihood)
    |   - Weighted heuristic combining:
    |     - Aspect ratio similarity to typical face (0.75 w/h)
    |     - Region compactness
    |     - Skin pixel density
    |     - Region area
    v
Detection Output
    |
    +---> draw_grid()  — Annotated image with bounding boxes
    +---> get_stats()  — Summary statistics
    +---> export()     — JSON or CSV output
```

## Module Responsibilities

### `core.py`
- `FaceGrid` class: orchestrates the detection pipeline
- `Detection` model: Pydantic model for structured detection results
- Methods: `load_image`, `detect_skin_regions`, `find_candidate_regions`, `analyze_region`, `score_face_likelihood`, `detect_faces`, `draw_grid`, `get_stats`, `export`

### `config.py`
- `FaceGridConfig`: Pydantic settings model
- HSV thresholds, grid size, scoring parameters, display settings

### `utils.py`
- `rgb_to_hsv()`: Pure numpy RGB-to-HSV conversion
- `skin_mask_from_hsv()`: Dual-range skin detection
- `compute_grid_skin_ratios()`: Grid overlay analysis
- `flood_fill_regions()`: Connected component discovery
- `bounding_box()`, `aspect_ratio()`: Geometry helpers
- `grid_cell_to_pixel_box()`: Coordinate mapping

## Design Decisions

1. **No ML models**: Detection uses only color and geometry heuristics, keeping the library lightweight and dependency-free beyond Pillow and numpy.

2. **Grid-based analysis**: Rather than pixel-level connected components (which are expensive), we reduce the problem to an NxN grid where N is configurable (default 16).

3. **Dual HSV ranges**: Human skin spans hue values near 0/360, requiring two HSV ranges to handle the wrap-around.

4. **Pydantic models**: Configuration and detection results use Pydantic for validation, serialization, and documentation.

5. **Scoring heuristic**: The weighted combination of aspect ratio, compactness, density, and area provides a reasonable face-likelihood estimate without training data.
