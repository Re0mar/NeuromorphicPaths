"""
A fake Pupil Labs client, for the tests that drive the live receiver and the capture script.

install() puts fake classes in place of the client's Device, its RTSP streamers and its Time Echo
estimator, on the client's own modules. Code that imports them when it runs, as the receiver and
the capture script both do, then gets the fakes. Each fake follows a FakeNeonScript: what each
stream sends, how it ends, and whether Time Echo answers.

The client's modules have to exist to be patched, so tests using this importorskip the client.
"""

# Standard library imports
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

DEFAULT_PARAMETER_SETS = [b"\x00\x00\x00\x01\x67sequence", b"\x00\x00\x00\x01\x68picture"]
CAMERA_MATRIX = [[891.0, 0.0, 807.0], [0.0, 890.0, 608.0], [0.0, 0.0, 1.0]]
DISTORTION = [-0.13, 0.11, 0.0, 0.0, 0.0, 0.17, 0.05, 0.03]


@dataclass
class FakeNeonScript:
    """What the fake glasses do. Every stream ends after its samples unless told otherwise."""

    scene_packets: list[tuple[float, bytes]] = field(default_factory=list)  # (stamp, payload)
    gaze_samples: list[tuple[float, float, float]] = field(default_factory=list)  # (stamp, x, y)
    imu_samples: list[tuple[float, float, float, float, float]] = field(default_factory=list)  # (stamp, w, x, y, z)
    # Raised by a stream once its samples are sent. None ends it normally.
    gaze_failure: BaseException | None = None
    imu_failure: BaseException | None = None
    # An IMU stream that sends nothing and never ends, the way a stalled sensor looks.
    imu_stays_silent: bool = False
    sample_interval_seconds: float = 0.01
    world_connected: bool = True
    # How many scene packets arrive before the stream description is readable. Live it is never the first.
    packets_before_description: int = 1
    parameter_sets: list[bytes] = field(default_factory=lambda: list(DEFAULT_PARAMETER_SETS))
    time_echo_port: int | None = 9999
    time_echo_never_answers: bool = False
    time_offset_ms: float = 1300.0
    round_trip_ms: float = 7.0
    # What discovery finds when no address is given. None means no glasses on the network.
    discovered_address: str | None = "192.0.2.9"


def install(script: FakeNeonScript, set_attribute: Callable[[object, str, object], None]) -> None:
    """
    Replace the client's classes with fakes following `script`.

    :param set_attribute: monkeypatch.setattr in a test, so the client is restored afterwards.
        Plain setattr in a spawned child, which ends with the test.
    """
    # Optional dependency, importorskipped by every caller. Imported here, because this module is
    # also imported by name in a spawned child, before it knows which test it is serving.
    import pupil_labs.realtime_api as client
    from pupil_labs.realtime_api import simple, time_echo
    from pupil_labs.realtime_api.streaming import base, gaze, imu, video
    from pupil_labs.realtime_api.streaming.base import SDPDataNotAvailableError

    def fake_discover_one_device(max_search_duration_seconds: float = 10.0) -> SimpleNamespace | None:
        # Blocking, like the real one, which runs an event loop of its own. Run here too, because
        # called from the receiver's loop it raises RuntimeError, and that is what the discovery
        # test is there to catch. Without this line the test passed with the crash put back.
        asyncio.run(asyncio.sleep(0))
        if script.discovered_address is None:
            return None
        return SimpleNamespace(address=script.discovered_address, close=lambda: None)

    status = SimpleNamespace(
        phone=SimpleNamespace(ip="192.0.2.7", time_echo_port=script.time_echo_port),
        direct_world_sensor=lambda: SimpleNamespace(url="rtsp://world", connected=script.world_connected),
        direct_gaze_sensor=lambda: SimpleNamespace(url="rtsp://gaze", connected=True),
        direct_imu_sensor=lambda: SimpleNamespace(url="rtsp://imu", connected=True),
    )
    calibration = SimpleNamespace(scene_camera_matrix=CAMERA_MATRIX, scene_distortion_coefficients=DISTORTION)

    class FakeDevice:
        def __init__(self, address: str, port: int) -> None:
            self.address = address

        async def __aenter__(self) -> "FakeDevice":
            return self

        async def __aexit__(self, *exception: object) -> None:
            return None

        async def get_status(self) -> SimpleNamespace:
            return status

        async def get_calibration(self) -> SimpleNamespace:
            return calibration

    class FakeStreamer:
        """The async context manager shape every client streamer has."""

        def __init__(self, url: str, *arguments: object, **options: object) -> None:
            self.url = url
            self.packets_sent = 0

        async def __aenter__(self) -> "FakeStreamer":
            return self

        async def __aexit__(self, *exception: object) -> None:
            return None

    class FakeVideoStreamer(FakeStreamer):
        def _require_description(self) -> None:
            if self.packets_sent <= script.packets_before_description:
                raise SDPDataNotAvailableError("SDP data not available in session")

        @property
        def encoding(self) -> str:
            self._require_description()
            return "h264"

        @property
        def sprop_parameter_set_payloads(self) -> list[bytes]:
            self._require_description()
            return list(script.parameter_sets)

    async def receive_raw(streamer: FakeVideoStreamer):
        for stamp, raw in script.scene_packets:
            await asyncio.sleep(script.sample_interval_seconds)
            streamer.packets_sent += 1
            yield SimpleNamespace(raw=raw, timestamp_unix_seconds=stamp)

    class FakeRawStreamer:
        # Called on the class with the video streamer as its argument, as the receiver does.
        receive = staticmethod(receive_raw)

    class FakeGazeStreamer(FakeStreamer):
        async def receive(self):
            for stamp, x, y in script.gaze_samples:
                await asyncio.sleep(script.sample_interval_seconds)
                yield SimpleNamespace(x=x, y=y, timestamp_unix_seconds=stamp)
            if script.gaze_failure is not None:
                raise script.gaze_failure

    class FakeImuStreamer(FakeStreamer):
        async def receive(self):
            if script.imu_stays_silent:
                await asyncio.Event().wait()
            for stamp, w, x, y, z in script.imu_samples:
                await asyncio.sleep(script.sample_interval_seconds)
                yield SimpleNamespace(quaternion=SimpleNamespace(w=w, x=x, y=y, z=z), timestamp_unix_seconds=stamp)
            if script.imu_failure is not None:
                raise script.imu_failure

    class FakeTimeOffsetEstimator:
        def __init__(self, address: str, port: int) -> None:
            self.address = address

        async def estimate(self, number_of_measurements: int = 100) -> SimpleNamespace:
            if script.time_echo_never_answers:
                await asyncio.Event().wait()
            return SimpleNamespace(
                time_offset_ms=SimpleNamespace(median=script.time_offset_ms),
                roundtrip_duration_ms=SimpleNamespace(median=script.round_trip_ms),
            )

    set_attribute(client, "Device", FakeDevice)
    set_attribute(base, "RTSPRawStreamer", FakeRawStreamer)
    set_attribute(video, "RTSPVideoFrameStreamer", FakeVideoStreamer)
    set_attribute(gaze, "RTSPGazeStreamer", FakeGazeStreamer)
    set_attribute(imu, "RTSPImuStreamer", FakeImuStreamer)
    set_attribute(time_echo, "TimeOffsetEstimator", FakeTimeOffsetEstimator)
    set_attribute(simple, "discover_one_device", fake_discover_one_device)
