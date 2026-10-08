"""
How much each group's clearance has been wobbling, which is what makes a reading surprising.

N is the standard deviation of a group's clearance over a short window. A post two meters away
whose measured distance jitters by a centimeter is not surprising. One whose distance jumps by
half a meter between frames is, and the planner treats it that way. The paper defines N as the
standard deviation of the measured quantity, in the same units as S.
"""

# Standard library imports
from collections import deque
from dataclasses import dataclass

# Third party imports
import numpy as np


@dataclass(frozen=True)
class ClearanceSample:
    timestamp_seconds: float
    clearance_meters: float
    centroid: np.ndarray  # (2,) lateral, forward, in whatever frame the scene is working in


class ClearanceHistory:
    """A short window of clearance samples per group."""

    def __init__(self, window_seconds: float, min_samples: int, noise_floor_meters: float) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window must be positive, got {window_seconds}")
        if min_samples < 2:
            # A standard deviation of one sample is undefined, and ddof=1 needs two.
            raise ValueError(f"min_samples must be at least 2, got {min_samples}")
        if noise_floor_meters <= 0:
            raise ValueError(f"noise floor must be positive, got {noise_floor_meters}")
        self._window_seconds = window_seconds
        self._min_samples = min_samples
        self._noise_floor_meters = noise_floor_meters
        self._samples: dict[int, deque[ClearanceSample]] = {}

    def update(self, timestamp_seconds: float, group_ids: list[int], clearances: list[float], centroids: list[np.ndarray]) -> None:
        """
        Record this frame's clearance for every group seen, and evict samples older than the window.

        :param timestamp_seconds: This frame's time.
        :param group_ids: Groups seen this frame.
        :param clearances: Their clearances, same order.
        :param centroids: Their (lateral, forward) centroids, same order.
        """
        if not len(group_ids) == len(clearances) == len(centroids):
            raise ValueError("group_ids, clearances and centroids must line up")
        cutoff = timestamp_seconds - self._window_seconds
        for group_id, clearance_meters, centroid in zip(group_ids, clearances, centroids, strict=True):
            samples = self._samples.setdefault(group_id, deque())
            samples.append(ClearanceSample(timestamp_seconds, float(clearance_meters), np.asarray(centroid, dtype=np.float64)))
            while samples and samples[0].timestamp_seconds < cutoff:
                samples.popleft()

    def noise_scale(self, group_id: int) -> float:
        """
        N for a group: the sample standard deviation of its clearance over the window.

        Reports the floor when there are too few samples to say anything, and never less than
        the floor. N is the numerator of the surprise, so a group read as perfectly steady would
        otherwise cost nothing at all however close it is.
        """
        samples = self._samples.get(group_id)
        if samples is None or len(samples) < self._min_samples:
            return self._noise_floor_meters
        deviation = float(np.std([sample.clearance_meters for sample in samples], ddof=1))
        return max(deviation, self._noise_floor_meters)

    def closing_rate(self, group_id: int) -> float | None:
        """
        Meters per second the clearance shrank between the two most recent samples.

        Positive when the group is getting closer. None with fewer than two samples or no time
        between them. Carried on every point for the evaluation and a future motion term. The
        planner doesn't read it: its time to contact uses walking speed, and alarm.py says why.
        """
        samples = self._samples.get(group_id)
        if samples is None or len(samples) < 2:
            return None
        newer, older = samples[-1], samples[-2]
        gap = newer.timestamp_seconds - older.timestamp_seconds
        if gap <= 0:
            return None
        return (older.clearance_meters - newer.clearance_meters) / gap

    def velocity(self, group_id: int) -> np.ndarray | None:
        """
        Ground velocity (lateral, forward) of the group's centroid between the two most recent samples.

        Only meaningful when the samples share a frame that does not move with the walker. The
        pipeline asks for this only in the world frame.
        """
        samples = self._samples.get(group_id)
        if samples is None or len(samples) < 2:
            return None
        newer, older = samples[-1], samples[-2]
        gap = newer.timestamp_seconds - older.timestamp_seconds
        if gap <= 0:
            return None
        return (newer.centroid - older.centroid) / gap

    def forget_unseen(self, timestamp_seconds: float) -> None:
        """Drop groups with no sample inside the window, so ids from long ago do not pile up."""
        cutoff = timestamp_seconds - self._window_seconds
        stale = [group_id for group_id, samples in self._samples.items() if not samples or samples[-1].timestamp_seconds < cutoff]
        for group_id in stale:
            del self._samples[group_id]

    def tracked_group_ids(self) -> list[int]:
        return sorted(self._samples)
