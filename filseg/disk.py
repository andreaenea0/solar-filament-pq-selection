"""Measure the solar limb and brightness in full-disk display JPEGs."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class Disk:
    """Fitted centre (pixel x, y), radius and diagnostics."""

    x: float
    y: float
    radius: float
    rms: float
    rays: int


def _fit_edges(drop, x, y, cx, cy):
    """Fit and trim a circle from one supported drop per radial ray."""
    edge = drop.argmax(axis=1)
    rays = np.arange(len(drop))
    strength = drop[rays, edge]
    keep = (strength >= max(2, np.median(strength) / 3)) & (edge > 0) & (edge < drop.shape[1] - 1)
    px = (x[rays, edge] + x[rays, edge + 1]) / 2 - cx
    py = (y[rays, edge] + y[rays, edge + 1]) / 2 - cy
    design = np.column_stack((2 * px, 2 * py, np.ones(len(px))))
    for iteration in range(4):
        if np.min(keep.reshape(8, 45).sum(axis=1)) < 10:
            raise ValueError("Limb edges do not cover the full disk.")
        dx, dy, constant = np.linalg.lstsq(design[keep], (px**2 + py**2)[keep], rcond=None)[0]
        fitted = np.sqrt(constant + dx**2 + dy**2)
        residual = np.hypot(px - dx, py - dy) - fitted
        if iteration < 3:
            mad = np.median(np.abs(residual[keep] - np.median(residual[keep])))
            keep &= np.abs(residual) <= max(2, 4 * mad)
    return Disk(cx + dx, cy + dy, fitted, np.sqrt(np.mean(residual[keep] ** 2)), int(keep.sum()))


def fit_limb(image):
    """Fit the standardized GONG limb, with a wider search for other disks."""
    h, w = image.shape
    cx, cy = (w - 1) / 2, (h - 1) / 2
    scale = min(h, w)
    radius = np.arange(int(0.34 * scale), int(0.51 * scale), dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    x = (cx + np.cos(angles[:, None]) * radius).astype(np.float32)
    y = (cy + np.sin(angles[:, None]) * radius).astype(np.float32)
    samples = cv2.remap(image.astype(np.float32), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    drop = -np.diff(gaussian_filter1d(samples, sigma=1.5, axis=1), axis=1)

    # GONG remaps the disk to a nominal radius of 900 px at 2048 px.
    # Search this band first so a stronger outer halo cannot win by contrast.
    start, stop = np.searchsorted(radius, scale * np.array([0.432, 0.448]))
    try:
        return _fit_edges(drop[:, start:stop], x[:, start:stop + 1], y[:, start:stop + 1], cx, cy)
    except ValueError:
        return _fit_edges(drop, x, y, cx, cy)


def profile(image, disk, stride=4):
    """Return aperture contamination and sampled median JPEG brightness in annuli."""
    h, w = image.shape
    yy, xx = np.ogrid[:h:stride, :w:stride]
    radius = np.hypot(xx - disk.x, yy - disk.y)
    nominal = np.hypot(xx - (w - 1) / 2, yy - (h - 1) / 2)
    pixels = image[::stride, ::stride]
    out = {}
    for aperture in (900, 950):
        inside = nominal <= aperture
        out[f"r{aperture}_off_disk_fraction"] = float((radius[inside] > disk.radius).mean())
        out[f"r{aperture}_disk_excluded_fraction"] = float((nominal[radius <= disk.radius] > aperture).mean())
    relative = radius / disk.radius
    for low, high in ((0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .9), (.9, .95), (.95, .98)):
        values = pixels[(relative >= low) & (relative < high)]
        out[f"median_{low:g}_{high:g}"] = float(np.median(values))
    return out
