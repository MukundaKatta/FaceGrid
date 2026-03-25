"""Configuration for FaceGrid face detection parameters."""

from __future__ import annotations

from typing import Tuple

from pydantic import BaseModel, Field


class FaceGridConfig(BaseModel):
    """Configuration for the FaceGrid detector.

    Controls grid resolution, skin color thresholds in HSV space,
    and scoring parameters for face-likelihood analysis.
    """

    grid_size: int = Field(
        default=16,
        ge=4,
        le=128,
        description="Number of cells per row and column in the analysis grid.",
    )

    min_skin_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of skin-colored pixels in a cell to flag it.",
    )

    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum face-likelihood score to include a detection.",
    )

    hsv_lower: Tuple[int, int, int] = Field(
        default=(0, 30, 60),
        description="Lower HSV bound for skin color detection (H, S, V).",
    )

    hsv_upper: Tuple[int, int, int] = Field(
        default=(25, 180, 255),
        description="Upper HSV bound for skin color detection (H, S, V).",
    )

    hsv_lower_alt: Tuple[int, int, int] = Field(
        default=(160, 30, 60),
        description="Alternate lower HSV bound to capture reddish skin tones wrapping around hue.",
    )

    hsv_upper_alt: Tuple[int, int, int] = Field(
        default=(180, 180, 255),
        description="Alternate upper HSV bound for reddish skin tones.",
    )

    min_region_area: int = Field(
        default=4,
        ge=1,
        description="Minimum number of contiguous grid cells for a candidate region.",
    )

    max_aspect_ratio: float = Field(
        default=2.0,
        gt=0.0,
        description="Maximum aspect ratio deviation for face-like bounding boxes.",
    )

    border_color: Tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="RGB color for drawing detection bounding boxes.",
    )

    border_width: int = Field(
        default=2,
        ge=1,
        description="Line width for detection bounding boxes.",
    )
