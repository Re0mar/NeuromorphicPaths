"""
Checks what units Neon Player's depth plugin saved, by comparing its saved maps with the model's own output.

Run by hand, with the glasses extra and a GPU, on a recording the plugin has already run over:

    python examples/compare_plugin_cache.py frame_logs/recordings/walk_2026_10_08_b

Depth Anything 3's metric model answers as if every camera had a 300 px focal. The plugin multiplies
by the recording's focal, scaled to the width the model ran at, over 300, before it saves. This runs
the same checkpoint on the same frames at the plugin's 504 px, resizes the raw output the way the
plugin does, and divides the saved maps by it. The printout compares that ratio with what one
conversion, none and two would give.

On walk_2026_10_08_b it measured 0.9343 against 0.9353 for one conversion, on 10 frames.
"""

# Standard library imports
import argparse
import os
import sys
from pathlib import Path

# Third party imports
import cv2
import numpy as np

# Local package imports
from nav.sources.neon_recording import NativeNeonRecordingReader

# The plugin's own cache path and file name for the metric model, at pupil-labs/npp-depth-estimation e6202a9.
CACHE_FILE = Path(".neon_player") / "cache" / "DepthEstimationPlugin" / "depth_values_DA3Metric-Large.npy"
# The plugin calls the model with its default processing resolution.
PLUGIN_PROCESS_RESOLUTION = 504
CANONICAL_FOCAL_PIXELS = 300.0


def one_conversion_ratio(focal_pixels: float, network_width: int, scene_width: int) -> float:
    """
    What the plugin's saved values are over the model's raw output, when it converted once.

    :param focal_pixels: The recording's mean focal, at the scene's size.
    :param network_width: The width of the depth map the model returned.
    :param scene_width: The scene video's width.
    :return: The network focal over 300.
    :rtype: float
    """
    return focal_pixels * network_width / scene_width / CANONICAL_FOCAL_PIXELS


def saved_over_raw(saved_map: np.ndarray, raw_depth: np.ndarray, scene_size: tuple[int, int]) -> float:
    """
    The median, over pixels, of one saved map divided by the raw output resized the plugin's way.

    The plugin resizes to the scene's full size, then down to a quarter with INTER_AREA, so the raw
    output takes the same two steps before the division.

    :param saved_map: (rows, columns) as the plugin saved it, a quarter of the scene size.
    :param raw_depth: The model's output for the same frame, at the network's size.
    :param scene_size: (height, width) of the scene video.
    :return: The median ratio.
    :rtype: float
    """
    height, width = scene_size
    full = cv2.resize(np.asarray(raw_depth, dtype=np.float32), (width, height))
    quarter = cv2.resize(full, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
    if quarter.shape != saved_map.shape:
        raise ValueError(f"the saved map is {saved_map.shape}, the resized output {quarter.shape}")
    return float(np.median(np.asarray(saved_map, dtype=np.float64) / quarter))


def _positive_int(text: str) -> int:
    """A frame count above zero. np.linspace takes 0 and negatives, and the run would print nothing."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be above zero, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the units of Neon Player's depth plugin cache.")
    parser.add_argument("recording_dir", type=Path, help="a native Neon recording the plugin has run over")
    parser.add_argument("--frames", type=_positive_int, default=10, help="how many frames, spread evenly through the recording")
    arguments = parser.parse_args(argv)

    cache_path = arguments.recording_dir / CACHE_FILE
    if not cache_path.is_file():
        print(f"no plugin cache at {cache_path}", file=sys.stderr)
        return 1

    # The accelerated downloader isn't always installed and falls over on partial downloads, as in
    # nav/sources/estimator.py. Set before the model import pulls in the Hugging Face client.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    # Heavy C extensions, deferred so the arithmetic above imports and tests without them.
    import torch
    from depth_anything_3.api import DepthAnything3

    saved = np.load(cache_path, mmap_mode="r")
    reader = NativeNeonRecordingReader(arguments.recording_dir)
    try:
        matrix = reader.scene_camera_matrix()
        focal = (matrix[0, 0] + matrix[1, 1]) / 2.0
        scene_size = reader.scene_size()
        model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda" if torch.cuda.is_available() else "cpu").eval()

        ratios = []
        network_width = 0
        with torch.no_grad():
            for index in np.linspace(0, saved.shape[0] - 1, arguments.frames).round().astype(int).tolist():
                # Unstraightened, as the plugin read it.
                raw = np.asarray(model.inference([reader.scene_frame_rgb(index)], process_res=PLUGIN_PROCESS_RESOLUTION).depth[0])
                network_width = raw.shape[1]
                ratios.append(saved_over_raw(saved[index], raw, scene_size))
                print(f"frame {index:4d}: saved over raw, median per pixel {ratios[-1]:.4f}")
    finally:
        # The model load or a frame can fail, and the recording's video files stay open otherwise.
        reader.close()

    once = one_conversion_ratio(focal, network_width, scene_size[1])
    print(f"focal {focal:.3f} px, network width {network_width}: one conversion {once:.4f}, none 1.0000, two {once * once:.4f}")
    print(f"measured median {np.median(ratios):.4f}, from {min(ratios):.4f} to {max(ratios):.4f}, over {len(ratios)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
