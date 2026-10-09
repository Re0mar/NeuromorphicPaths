"""
The smallest path, view and field the web sink can send, and a stand-in for the glasses and demo
switch, shared by the sink tests and the browser tests.

Both suites build the same messages, so a key the page reads is the key the sink sends, and a
test on either side that changes the shape changes it for both.
"""

# Third party imports
import numpy as np

# Local package imports
from nav.sources.config import SourceMode
from nav.sources.switching import GlassesLink, SwitchState
from nav.types import DebugView, DepthFrame, FloorSource, ObstacleSet, Plane, PlannedPath, Pose

VIEW_SIDE = 4


def sample_path(heading: float = 0.1, alarm: bool = False, ear_gain_left: float = 1.0, ear_gain_right: float = 1.0) -> PlannedPath:
    """A two-step path with the given heading, alarm and ear gains, and nothing else of note."""
    return PlannedPath(
        1.0,
        np.array([0.0, 0.1]),
        np.array([0.0, 0.05]),
        heading,
        alarm,
        2.5,
        scene_information_bits=0.0,
        avoidance_surprise_bits=0.0,
        alarm_pan=0.0 if alarm else None,
        ear_gain_left=ear_gain_left,
        ear_gain_right=ear_gain_right,
    )


def sample_view() -> DebugView:
    """A 4 by 4 frame of valid depth, no obstacles, a level floor. Enough to render and encode."""
    frame = DepthFrame(
        timestamp_seconds=1.0,
        depth_meters=np.full((VIEW_SIDE, VIEW_SIDE), 2.0, dtype=np.float32),
        intrinsics=np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]]),
        pose=Pose(np.array([1.0, 0.0, 0.0, 0.0]), None, False),
        ground_plane=None,
        gaze_pixel=None,
    )
    return DebugView(frame, ObstacleSet(1.0, (), 0), Plane(np.array([0.0, -1.0, 0.0]), 1.6), FloorSource.FITTED, 1.4, 0.30, 1.47)


def sample_field() -> tuple[np.ndarray, np.ndarray]:
    """A field of two steps by three cells, all zero, and the grid under it."""
    return np.zeros((2, 3)), np.array([-1.0, 0.0, 1.0])


class FakeSwitch:
    """The glasses and demo switch as the sink sees it: state to subscribe to, and requests to make."""

    def __init__(self, mode: SourceMode = SourceMode.GLASSES, glasses: GlassesLink = GlassesLink.CONNECTED, detail: str | None = None) -> None:
        self.requests: list[SourceMode] = []
        self.listeners: list = []
        self.current = SwitchState(mode, glasses, detail)

    def subscribe(self, listener):
        self.listeners.append(listener)
        listener(self.current)
        return lambda: self.listeners.remove(listener)

    def request(self, mode: SourceMode) -> None:
        self.requests.append(mode)
        self.current = SwitchState(mode, self.current.glasses, self.current.glasses_detail)
        for listener in list(self.listeners):
            listener(self.current)
