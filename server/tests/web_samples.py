"""
The smallest path, view and field the web sink can send, and a stand-in for the switch
between the live input and a recording, shared by the sink tests and the browser tests.

Both suites build the same messages, so a key the page reads is the key the sink sends, and a
test on either side that changes the shape changes it for both.
"""

# Standard library imports
import dataclasses

# Third party imports
import numpy as np

# Local package imports
from nav.planner.alarm import avoidance_surprise_bits_at
from nav.sources.config import SourceMode
from nav.sources.switching import InputLink, SwitchState
from nav.types import DebugView, DepthFrame, FloorSource, ObstacleSet, Plane, PlannedPath, Pose

VIEW_SIDE = 4


def sample_path(
    heading: float = 0.1,
    alarm: bool = False,
    ear_gain_left: float = 1.0,
    ear_gain_right: float = 1.0,
    time_to_contact_seconds: float | None = None,
    danger_noticed: bool = False,
    alarm_audible: bool | None = None,
) -> PlannedPath:
    """
    A two-step path with the given heading, alarm, ear gains and time to contact, and nothing else of note.

    :param time_to_contact_seconds: Sent as the avoidance surprise it stands for, since the wire
        derives the time from the surprise. None is 0 bits, an empty corridor.
    :param danger_noticed: Whether the eyes are on the danger.
    :param alarm_audible: Whether the alarm is heard. None is the same as alarm, as every path before the gaze switch.
    """
    return PlannedPath(
        1.0,
        np.array([0.0, 0.1]),
        np.array([0.0, 0.05]),
        heading,
        alarm,
        2.5,
        scene_information_bits=0.0,
        avoidance_surprise_bits=0.0 if time_to_contact_seconds is None else avoidance_surprise_bits_at(time_to_contact_seconds),
        alarm_pan=0.0 if alarm else None,
        ear_gain_left=ear_gain_left,
        ear_gain_right=ear_gain_right,
        danger_noticed=danger_noticed,
        alarm_audible=alarm_audible,
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
    """The live and recording switch as the sink sees it: state to subscribe to, and requests to make."""

    def __init__(
        self,
        mode: SourceMode = SourceMode.LIVE,
        live: InputLink = InputLink.CONNECTED,
        detail: str | None = None,
        has_recording: bool = True,
        live_has_video: bool = True,
        recordings: tuple[str, ...] | None = None,
    ) -> None:
        self.requests: list[SourceMode] = []
        self.recording_requests: list[str] = []
        self.listeners: list = []
        if recordings is None:
            recordings = ("walk",) if has_recording else ()
        self.current = SwitchState(mode, live, detail, has_recording, live_has_video, recordings, recordings[0] if recordings else None)

    def subscribe(self, listener):
        self.listeners.append(listener)
        listener(self.current)
        return lambda: self.listeners.remove(listener)

    def request(self, mode: SourceMode) -> None:
        self.requests.append(mode)
        self.announce(dataclasses.replace(self.current, mode=mode))

    def request_recording(self, name: str) -> None:
        self.recording_requests.append(name)
        self.announce(dataclasses.replace(self.current, recording=name))

    def announce(self, state: SwitchState) -> None:
        """Tell every listener of a new state, the way the real switch does when the live link changes."""
        self.current = state
        for listener in list(self.listeners):
            listener(state)
