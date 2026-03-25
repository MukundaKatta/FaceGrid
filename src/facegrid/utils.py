"""Utility functions for color space conversion, region analysis, and bounding box helpers."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def rgb_to_hsv(rgb_array: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Convert an RGB image array to HSV color space.

    Args:
        rgb_array: Image array of shape (H, W, 3) with values in [0, 255].

    Returns:
        HSV array of shape (H, W, 3) where H is in [0, 180], S in [0, 255], V in [0, 255].
    """
    rgb_normalized = rgb_array.astype(np.float64) / 255.0

    r = rgb_normalized[:, :, 0]
    g = rgb_normalized[:, :, 1]
    b = rgb_normalized[:, :, 2]

    cmax = np.max(rgb_normalized, axis=2)
    cmin = np.min(rgb_normalized, axis=2)
    diff = cmax - cmin

    h = np.zeros_like(cmax)
    s = np.zeros_like(cmax)
    v = cmax

    # Saturation
    nonzero_mask = cmax > 0
    s[nonzero_mask] = diff[nonzero_mask] / cmax[nonzero_mask]

    # Hue calculation
    diff_nonzero = diff > 1e-10

    mask_r = diff_nonzero & (cmax == r)
    mask_g = diff_nonzero & (cmax == g) & ~mask_r
    mask_b = diff_nonzero & ~mask_r & ~mask_g

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)

    # Scale to OpenCV-like ranges: H [0, 180], S [0, 255], V [0, 255]
    hsv = np.zeros_like(rgb_normalized)
    hsv[:, :, 0] = h / 2.0  # 0-360 -> 0-180
    hsv[:, :, 1] = s * 255.0
    hsv[:, :, 2] = v * 255.0

    return hsv


def skin_mask_from_hsv(
    hsv: NDArray[np.float64],
    lower: Tuple[int, int, int],
    upper: Tuple[int, int, int],
    lower_alt: Tuple[int, int, int],
    upper_alt: Tuple[int, int, int],
) -> NDArray[np.bool_]:
    """Create a binary skin mask from an HSV image.

    Uses two HSV ranges to handle the hue wrap-around for skin tones.

    Args:
        hsv: HSV image array of shape (H, W, 3).
        lower: Lower HSV bound (primary range).
        upper: Upper HSV bound (primary range).
        lower_alt: Lower HSV bound (alternate range for hue wrap).
        upper_alt: Upper HSV bound (alternate range for hue wrap).

    Returns:
        Boolean mask of shape (H, W) where True indicates skin pixels.
    """
    lower_arr = np.array(lower, dtype=np.float64)
    upper_arr = np.array(upper, dtype=np.float64)
    lower_alt_arr = np.array(lower_alt, dtype=np.float64)
    upper_alt_arr = np.array(upper_alt, dtype=np.float64)

    mask_primary = np.all((hsv >= lower_arr) & (hsv <= upper_arr), axis=2)
    mask_alt = np.all((hsv >= lower_alt_arr) & (hsv <= upper_alt_arr), axis=2)

    return mask_primary | mask_alt


def compute_grid_skin_ratios(
    skin_mask: NDArray[np.bool_], grid_size: int
) -> NDArray[np.float64]:
    """Compute the skin pixel ratio for each cell in a grid overlay.

    Args:
        skin_mask: Boolean mask of shape (H, W).
        grid_size: Number of divisions along each axis.

    Returns:
        Array of shape (grid_size, grid_size) with skin ratios per cell.
    """
    h, w = skin_mask.shape
    cell_h = h / grid_size
    cell_w = w / grid_size

    ratios = np.zeros((grid_size, grid_size), dtype=np.float64)

    for row in range(grid_size):
        y0 = int(round(row * cell_h))
        y1 = int(round((row + 1) * cell_h))
        for col in range(grid_size):
            x0 = int(round(col * cell_w))
            x1 = int(round((col + 1) * cell_w))

            cell = skin_mask[y0:y1, x0:x1]
            if cell.size > 0:
                ratios[row, col] = np.mean(cell)

    return ratios


def flood_fill_regions(
    grid: NDArray[np.bool_],
) -> List[List[Tuple[int, int]]]:
    """Find connected regions in a boolean grid using flood fill.

    Args:
        grid: Boolean grid of shape (rows, cols).

    Returns:
        List of regions, each a list of (row, col) cell coordinates.
    """
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    regions: List[List[Tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] and not visited[r, c]:
                region: List[Tuple[int, int]] = []
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc] or not grid[cr, cc]:
                        continue
                    visited[cr, cc] = True
                    region.append((cr, cc))
                    stack.extend([(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)])
                if region:
                    regions.append(region)

    return regions


def bounding_box(cells: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Compute the bounding box of a list of grid cell coordinates.

    Args:
        cells: List of (row, col) coordinates.

    Returns:
        Tuple of (min_row, min_col, max_row, max_col).
    """
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def aspect_ratio(min_row: int, min_col: int, max_row: int, max_col: int) -> float:
    """Compute the aspect ratio (width / height) of a bounding box.

    Args:
        min_row: Top row index.
        min_col: Left column index.
        max_row: Bottom row index.
        max_col: Right column index.

    Returns:
        Aspect ratio as a float. Returns 1.0 for degenerate boxes.
    """
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    if height == 0:
        return 1.0
    return width / height


def grid_cell_to_pixel_box(
    min_row: int,
    min_col: int,
    max_row: int,
    max_col: int,
    image_width: int,
    image_height: int,
    grid_size: int,
) -> Tuple[int, int, int, int]:
    """Convert grid cell bounding box to pixel coordinates.

    Args:
        min_row, min_col, max_row, max_col: Grid cell bounding box.
        image_width: Width of the original image in pixels.
        image_height: Height of the original image in pixels.
        grid_size: Number of grid divisions per axis.

    Returns:
        Tuple of (x0, y0, x1, y1) in pixel coordinates.
    """
    cell_w = image_width / grid_size
    cell_h = image_height / grid_size

    x0 = int(round(min_col * cell_w))
    y0 = int(round(min_row * cell_h))
    x1 = int(round((max_col + 1) * cell_w))
    y1 = int(round((max_row + 1) * cell_h))

    return x0, y0, x1, y1
