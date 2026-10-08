"""
Facts about the Neon's scene camera that both Neon sources need.

Kept out of neon_live.py so the recording source can read them without importing the live client's
process machinery.
"""

# The scene camera's native size, which the device's calibration describes. The calibration buffer
# does not carry a size of its own.
NEON_SCENE_SIZE = (1200, 1600)  # height, width
