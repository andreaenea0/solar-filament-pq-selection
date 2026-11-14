"""Build probability-map channels with a fixed ensemble size per family."""

from collections.abc import Sequence

import numpy as np


def read_map(path):
    """Read a raw .npy map or the p array of an existing .npz export."""
    if str(path).endswith(".npy"):
        return np.load(path, allow_pickle=False)
    with np.load(path, allow_pickle=False) as archive:
        return archive["p"]


def channel_mean(maps):
    """Average channels as float32, preserving single-channel [H, W] maps."""
    maps = maps.astype(np.float32)
    return maps.mean(0) if maps.ndim == 3 else maps


def matched_family_maps(
    families: Sequence[Sequence[np.ndarray]], member_counts: Sequence[int], *, quantize: bool = True
) -> np.ndarray:
    """Average the first requested members of each family into uint8 [C, H, W].

    Use the same family order, counts and member recipe in OOF and test. Each
    input is one uint8 [H, W] map after D4 averaging. Order members in advance
    (e.g. by seed or fold), and only supply held-out maps for OOF images.
    For a single family, save channel 0 as a 2D map for tools/build_cands.py.
    Set quantize=False to retain float32 byte-scale means for the keep rule.
    """
    if len(families) == 0 or len(families) != len(member_counts):
        raise ValueError("Provide a member count for every map family.")

    channels = []
    shape = None
    for members, count in zip(families, member_counts):
        if not isinstance(count, (int, np.integer)) or not 1 <= count <= len(members):
            raise ValueError("Each family must have at least its requested number of members.")
        selected = members[:count]
        if shape is None:
            shape = selected[0].shape
        if any(p.ndim != 2 or p.shape != shape or p.dtype != np.uint8 for p in selected):
            raise ValueError("Member maps must be uint8 [H, W] with the same shape.")
        mean = np.mean(selected, axis=0, dtype=np.float32)
        channels.append(mean.round().astype(np.uint8) if quantize else mean)
    return np.stack(channels)
