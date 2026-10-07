"""Top-down map image with the driven route."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# Shared with the web viewer's height palette (viewer/index.html).
HEIGHT_PALETTE = np.array(
    [[29, 78, 137], [38, 166, 154], [117, 201, 80], [253, 210, 45], [224, 72, 50]], dtype=float
)
BACKGROUND = np.array([248, 250, 252], dtype=float)
ROUTE_COLOR = (39, 55, 77)
START_COLOR = (27, 153, 139)
END_COLOR = (209, 73, 91)


def palette_colors(normalized: np.ndarray) -> np.ndarray:
    """Interpolate [0, 1] values through the height palette into RGB floats."""
    positions = np.linspace(0.0, 1.0, len(HEIGHT_PALETTE))
    return np.stack(
        [np.interp(normalized, positions, HEIGHT_PALETTE[:, channel]) for channel in range(3)],
        axis=-1,
    )


def robust_range(values: np.ndarray) -> tuple[float, float]:
    """2nd-98th percentile range so a few outliers do not flatten the colors."""
    lower, upper = np.quantile(values, [0.02, 0.98])
    if np.isclose(lower, upper):
        lower, upper = lower - 0.5, upper + 0.5
    return float(lower), float(upper)


def rasterize_topdown(
    points: np.ndarray, resolution: float, colors: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a map from above: height-colored (or RGB) cells, density as opacity."""
    minimum_xy = np.floor(points[:, :2].min(axis=0) / resolution) * resolution
    maximum_xy = np.ceil(points[:, :2].max(axis=0) / resolution) * resolution
    width = max(1, int(np.ceil((maximum_xy[0] - minimum_xy[0]) / resolution)))
    height = max(1, int(np.ceil((maximum_xy[1] - minimum_xy[1]) / resolution)))

    x = np.minimum(((points[:, 0] - minimum_xy[0]) / resolution).astype(int), width - 1)
    y = np.minimum(((maximum_xy[1] - points[:, 1]) / resolution).astype(int), height - 1)
    cells = y * width + x
    counts = np.bincount(cells, minlength=width * height)
    occupied = counts > 0

    if colors is None:
        top = np.full(width * height, -np.inf)
        np.maximum.at(top, cells, points[:, 2])
        lower, upper = robust_range(top[occupied])
        cell_colors = palette_colors(np.clip((top[occupied] - lower) / (upper - lower), 0.0, 1.0))
    else:
        sums = np.stack([np.bincount(cells, colors[:, c] * 255.0, width * height) for c in range(3)], -1)
        cell_colors = sums[occupied] / counts[occupied, None]

    log_counts = np.log1p(counts[occupied])
    opacity = 0.3 + 0.7 * np.clip(log_counts / max(float(np.quantile(log_counts, 0.99)), 1.0), 0, 1)
    pixels = np.broadcast_to(BACKGROUND, (width * height, 3)).copy()
    pixels[occupied] = BACKGROUND * (1 - opacity[:, None]) + cell_colors * opacity[:, None]
    return pixels.reshape(height, width, 3).astype(np.uint8), minimum_xy, maximum_xy


def draw_route(
    image: Image.Image,
    trajectory: np.ndarray,
    minimum_xy: np.ndarray,
    maximum_y: float,
    scale: float,
) -> None:
    """Draw the XY route with start and end markers onto a top-down image."""
    pixels = [
        (float((x - minimum_xy[0]) * scale), float((maximum_y - y) * scale))
        for x, y in trajectory[:, :2]
    ]
    draw = ImageDraw.Draw(image)
    radius = max(3, image.width // 150)
    if len(pixels) > 1:
        draw.line(pixels, fill=ROUTE_COLOR, width=max(1, radius // 2))
    for (u, v), color in ((pixels[0], START_COLOR), (pixels[-1], END_COLOR)):
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color)


def save_topdown_map(
    points: np.ndarray,
    poses: Sequence[np.ndarray],
    destination: Path,
    resolution: float,
    colors: np.ndarray | None = None,
) -> None:
    """Write a meter-scaled top-down PNG of the whole map with the estimated route."""
    raster, minimum_xy, maximum_xy = rasterize_topdown(points, resolution, colors)
    image = Image.fromarray(raster)
    trajectory = np.asarray([pose[:3, 3] for pose in poses])
    draw_route(image, trajectory, minimum_xy, maximum_xy[1], 1.0 / resolution)
    image.save(destination, optimize=True)
