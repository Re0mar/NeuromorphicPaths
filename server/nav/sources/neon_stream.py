"""
The Neon's scene video, gaze and IMU, received and decoded our way, inside the device process.

The Pupil Labs client's simple Device converts every decoded frame to BGR, 30 a second, on the one
thread that also decodes. That thread sat at 94 % of a core on an idle laptop. With the head moving
and the pipeline running beside it, it fell behind by about a fifth of real time and the frames
reached the planner up to 5.6 s late. Here the decoder keeps only the newest frame and converts it
when the pipeline asks for one, about 3 times a second.

Packets come from one of two places, and everything after that is the same code:
    - the glasses, through the client's own RTSP readers, which do no decoding
    - a capture folder written by examples/capture_neon_stream.py, fed at the pace it was recorded

A played-back capture is stamped as if it were happening now, so the pipeline's latency figures
measure this laptop the same way they would live.

NeonStreamDevice answers the same calls the client's simple Device does, so the device process
serves either one.
"""

# Standard library imports
import asyncio
import base64
import bisect
import collections
import json
import logging
import queue
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.sources.config import NeonConfig

log = logging.getLogger(__name__)

# The capture format, shared with examples/capture_neon_stream.py.
PACKET_HEADER = struct.Struct("<dI")  # timestamp in seconds, payload length in bytes
SCENE_PACKETS_FILENAME = "scene_packets.bin"
GAZE_FILENAME = "gaze.jsonl"
IMU_FILENAME = "imu.jsonl"
META_FILENAME = "meta.json"

# How far a gaze sample may be from a frame and still count as that frame's gaze. A frame is 33 ms.
GAZE_MATCH_TOLERANCE_SECONDS = 0.05
GAZE_HISTORY = 400  # About two seconds of gaze at 200 Hz.
# Decoder threads. Frame threading would add a frame of delay per thread. Slice threading costs
# nothing, and helps whenever the phone's encoder splits a frame into slices.
DECODER_THREAD_TYPE = "SLICE"
# The decode thread checks for a stop this often when no packets come.
DECODE_POLL_SECONDS = 0.25
# Half a second of scene packets at about 600 a second. More than that waiting means the decoder is
# falling behind, which is the delay this module exists to prevent, so it is logged.
DECODE_BACKLOG_WARNING_PACKETS = 300
BACKLOG_WARNING_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True)
class StreamCalibration:
    scene_camera_matrix: np.ndarray
    scene_distortion_coefficients: np.ndarray


@dataclass(frozen=True)
class StreamEstimate:
    median: float


@dataclass(frozen=True)
class StreamTimeEcho:
    time_offset_ms: StreamEstimate
    roundtrip_duration_ms: StreamEstimate


@dataclass(frozen=True)
class StreamFrame:
    bgr_pixels: np.ndarray
    timestamp_unix_seconds: float


@dataclass(frozen=True)
class StreamGaze:
    x: float
    y: float


@dataclass(frozen=True)
class StreamMatched:
    frame: StreamFrame
    gaze: StreamGaze | None


@dataclass(frozen=True)
class StreamQuaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class StreamImuDatum:
    quaternion: StreamQuaternion | None


class SceneDecoder:
    """H.264 packets in, the newest decoded frame kept. Nothing converted until asked for."""

    def __init__(self, sprop_parameter_sets: list[bytes]) -> None:
        # Optional dependency. PyAV comes with the glasses extra, through the Pupil Labs client.
        import av

        # Same helper the client's own decoder uses. The packets arrive as RTP payloads.
        from pupil_labs.realtime_api.streaming.nal_unit import extract_payload_from_nal_unit

        self._extract = extract_payload_from_nal_unit
        self._codec = av.CodecContext.create("h264", "r")
        self._codec.thread_type = DECODER_THREAD_TYPE
        for parameter_set in sprop_parameter_sets:
            self._codec.parse(self._extract(parameter_set))
        self._frame_timestamp: float | None = None

    def feed(self, raw: bytes, timestamp_unix_seconds: float) -> tuple[object, float] | None:
        """
        One packet in. Returns the newest decoded frame and its stamp, if this packet completed one.

        A packet that starts a new frame flushes the previous one, which carries the previous
        packet's stamp. That is how the client does it, and why the stamp lags by one packet here.
        """
        newest = None
        for packet in self._codec.parse(self._extract(raw)):
            for frame in self._codec.decode(packet):
                if self._frame_timestamp is not None:
                    newest = (frame, self._frame_timestamp)
        self._frame_timestamp = timestamp_unix_seconds
        return newest


class NeonStreamDevice:
    """Receives on a background thread, and answers the simple Device's calls from what it kept."""

    def __init__(self, config: NeonConfig) -> None:
        self._config = config
        self._condition = threading.Condition()
        self._newest_frame: tuple[object, float] | None = None
        self._frames_decoded = 0
        self._frames_served = 0
        self._gaze: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=GAZE_HISTORY)
        self._imu: tuple[float, float, float, float, float] | None = None
        self._imu_served_stamp: float | None = None
        self._failure: BaseException | None = None
        self._replay_finished = False
        self._stopping = threading.Event()
        self._calibration: StreamCalibration | None = None
        self._time_offset: StreamTimeEcho | None = None
        self._replay_shift_seconds = 0.0
        # The address actually connected to, which is the discovered one when none was given.
        self._address: str | None = config.address
        self._thread = threading.Thread(target=self._run, name="neon-receive", daemon=True)
        self._packets: queue.Queue = queue.Queue()
        self._last_backlog_warning = 0.0
        self._decode_thread = threading.Thread(target=self._decode, name="neon-decode", daemon=True)

    def start(self) -> None:
        """Connect, read the calibration and the clock offset, and start receiving."""
        ready = threading.Event()
        self._ready = ready
        self._decode_thread.start()
        self._thread.start()
        if not ready.wait(self._config.discovery_timeout_seconds + 30.0):
            raise ConnectionError("the Neon stream did not start within the discovery timeout")
        if self._failure is not None:
            raise self._failure

    # The simple Device's calls. Shapes match the client's, field for field.

    def get_calibration(self) -> StreamCalibration:
        if self._calibration is None:
            raise ValueError("no calibration was read when the stream started")
        return self._calibration

    def estimate_time_offset(self, number_of_measurements: int = 100) -> StreamTimeEcho | None:
        if self._config.replay_dir is not None:
            return self._time_offset
        return _measure_time_offset(self._address, self._config.port, number_of_measurements)

    def receive_matched_scene_video_frame_and_gaze(self, timeout_seconds: float | None = None) -> StreamMatched | None:
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        with self._condition:
            # A frame the pipeline has not had yet. The newest one, whatever came before it.
            while self._frames_decoded == self._frames_served:
                self._raise_if_failed()
                if self._replay_finished:
                    # Every frame of the capture has been handed over. The run ends here, normally,
                    # rather than waiting for a frame that will never come.
                    raise EOFError("the capture has been played to the end")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            frame, timestamp = self._newest_frame
            self._frames_served = self._frames_decoded
            gaze = self._gaze_near(timestamp)
        # Converted here, once, for the one frame that is used.
        pixels = frame.to_ndarray(format="bgr24")
        return StreamMatched(StreamFrame(pixels, timestamp), gaze)

    def receive_imu_datum(self, timeout_seconds: float | None = None) -> StreamImuDatum | None:
        with self._condition:
            self._raise_if_failed()
            if self._imu is None or self._imu[0] == self._imu_served_stamp:
                return None
            stamp, w, x, y, z = self._imu
            self._imu_served_stamp = stamp
        return StreamImuDatum(StreamQuaternion(w, x, y, z))

    def close(self) -> None:
        self._stopping.set()
        self._thread.join(5.0)
        self._decode_thread.join(5.0)

    # The receiving side, all on the background thread.

    def _run(self) -> None:
        try:
            asyncio.run(self._receive())
        except Exception as failure:  # noqa: BLE001, handed to the caller's thread and raised there
            with self._condition:
                self._failure = failure
                self._condition.notify_all()
            log.error("the Neon stream stopped (caught %s): %s", type(failure).__name__, failure)
        finally:
            self._ready.set()

    async def _receive(self) -> None:
        if self._config.replay_dir is not None:
            await self._receive_replay(Path(self._config.replay_dir))
        else:
            await self._receive_live()

    async def _receive_live(self) -> None:
        # Optional dependency. Only present with the glasses extra.
        from pupil_labs.realtime_api import Device
        from pupil_labs.realtime_api.streaming.base import RTSPRawStreamer
        from pupil_labs.realtime_api.streaming.gaze import RTSPGazeStreamer
        from pupil_labs.realtime_api.streaming.imu import RTSPImuStreamer
        from pupil_labs.realtime_api.streaming.video import RTSPVideoFrameStreamer

        address = self._config.address or _discover(self._config)
        self._address = address
        async with Device(address, self._config.port) as device:
            status = await device.get_status()
            calibration = await device.get_calibration()
        self._calibration = StreamCalibration(
            np.asarray(calibration.scene_camera_matrix, dtype=np.float64),
            np.asarray(calibration.scene_distortion_coefficients, dtype=np.float64),
        )
        world, gaze, imu = status.direct_world_sensor(), status.direct_gaze_sensor(), status.direct_imu_sensor()
        if world is None or not world.connected:
            raise ConnectionError("the Neon's scene camera is not streaming. Is the Companion app open with the glasses plugged in?")
        self._ready.set()

        async def scene() -> None:
            async with RTSPVideoFrameStreamer(world.url) as streamer:
                decoder = None
                async for data in RTSPRawStreamer.receive(streamer):
                    if decoder is None:
                        try:
                            decoder = SceneDecoder([bytes(param) for param in streamer.sprop_parameter_set_payloads])
                        except Exception as not_yet:  # noqa: BLE001, the stream description arrives with the first packets
                            log.debug("waiting for the stream description (%s)", type(not_yet).__name__)
                            continue
                    self._on_packet(decoder, bytes(data.raw), data.timestamp_unix_seconds)
                    if self._stopping.is_set():
                        return

        async def gaze_stream() -> None:
            if gaze is None:
                return
            async with RTSPGazeStreamer(gaze.url) as streamer:
                async for datum in streamer.receive():
                    self._on_gaze(datum.timestamp_unix_seconds, float(datum.x), float(datum.y))
                    if self._stopping.is_set():
                        return

        async def imu_stream() -> None:
            if imu is None:
                return
            async with RTSPImuStreamer(imu.url) as streamer:
                async for datum in streamer.receive():
                    quaternion = datum.quaternion
                    self._on_imu(datum.timestamp_unix_seconds, quaternion.w, quaternion.x, quaternion.y, quaternion.z)
                    if self._stopping.is_set():
                        return

        await asyncio.gather(scene(), gaze_stream(), imu_stream())

    async def _receive_replay(self, capture: Path) -> None:
        meta = json.loads((capture / META_FILENAME).read_text(encoding="utf-8"))
        self._calibration = StreamCalibration(
            np.asarray(meta["scene_camera_matrix"], dtype=np.float64),
            np.asarray(meta["scene_distortion_coefficients"], dtype=np.float64),
        )
        offset = meta.get("time_offset")
        offset_ms = 0.0 if offset is None else float(offset["median_ms"])
        self._time_offset = StreamTimeEcho(StreamEstimate(offset_ms), StreamEstimate(0.0 if offset is None else float(offset["round_trip_median_ms"])))
        decoder = SceneDecoder([base64.b64decode(param) for param in meta["sprop"]])

        packets = list(_read_packets(capture / SCENE_PACKETS_FILENAME))
        gaze = _read_lines(capture / GAZE_FILENAME)
        imu = _read_lines(capture / IMU_FILENAME)
        if not packets:
            raise ValueError(f"{capture} holds no scene packets")
        first_stamp = packets[0][0]
        started = time.monotonic()
        # Moves the recording's Neon stamps to now. With the recorded offset added, a packet's
        # capture time on the laptop clock is the moment it is fed, so lag measures this laptop.
        self._replay_shift_seconds = (time.time() - offset_ms / 1000.0) - first_stamp
        log.info("replaying %d scene packets, %.0f s, from %s", len(packets), packets[-1][0] - first_stamp, capture.name)
        self._ready.set()

        events = sorted(
            [(stamp, "scene", payload) for stamp, payload in packets]
            + [(line["t"], "gaze", line) for line in gaze]
            + [(line["t"], "imu", line) for line in imu],
            key=lambda event: event[0],
        )
        for stamp, kind, payload in events:
            if self._stopping.is_set():
                return
            wait = (stamp - first_stamp) - (time.monotonic() - started)
            if wait > 0:
                await asyncio.sleep(wait)
            shifted = stamp + self._replay_shift_seconds
            if kind == "scene":
                self._on_packet(decoder, payload, shifted)
            elif kind == "gaze":
                self._on_gaze(shifted, payload["x"], payload["y"])
            else:
                self._on_imu(shifted, payload["w"], payload["x"], payload["y"], payload["z"])
        # Finished means decoded too, or the last frames would be lost to the end-of-replay check.
        await asyncio.to_thread(self._packets.join)
        log.info("replay finished")
        with self._condition:
            self._replay_finished = True
            self._condition.notify_all()

    def _on_packet(self, decoder: SceneDecoder, raw: bytes, timestamp: float) -> None:
        # Queued, not decoded here. This thread also parses every network packet in Python, and
        # doing both sat it at 99 % of a core and put frames seconds behind. The decode thread runs
        # alongside, because PyAV lets go of the GIL while it decodes.
        self._packets.put((decoder, raw, timestamp))
        waiting = self._packets.qsize()
        now = time.monotonic()
        if waiting > DECODE_BACKLOG_WARNING_PACKETS and now - self._last_backlog_warning > BACKLOG_WARNING_INTERVAL_SECONDS:
            log.warning("the scene decoder is %d packets behind", waiting)
            self._last_backlog_warning = now

    def _decode(self) -> None:
        """The decode thread: packets off the queue, the newest finished frame kept."""
        while not self._stopping.is_set():
            try:
                decoder, raw, timestamp = self._packets.get(timeout=DECODE_POLL_SECONDS)
            except queue.Empty:
                continue
            try:
                decoded = decoder.feed(raw, timestamp)
            finally:
                self._packets.task_done()
            if decoded is None:
                continue
            with self._condition:
                self._newest_frame = decoded
                self._frames_decoded += 1
                self._condition.notify_all()

    def _on_gaze(self, timestamp: float, x: float, y: float) -> None:
        with self._condition:
            self._gaze.append((timestamp, x, y))

    def _on_imu(self, timestamp: float, w: float, x: float, y: float, z: float) -> None:
        with self._condition:
            self._imu = (timestamp, w, x, y, z)

    def _gaze_near(self, timestamp: float) -> StreamGaze | None:
        """The gaze sample nearest the frame's stamp, within a frame and a half. Caller holds the lock."""
        if not self._gaze:
            return None
        stamps = [sample[0] for sample in self._gaze]
        index = bisect.bisect_left(stamps, timestamp)
        candidates = [self._gaze[i] for i in (index - 1, index) if 0 <= i < len(self._gaze)]
        nearest = min(candidates, key=lambda sample: abs(sample[0] - timestamp))
        if abs(nearest[0] - timestamp) > GAZE_MATCH_TOLERANCE_SECONDS:
            return None
        return StreamGaze(nearest[1], nearest[2])

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise ConnectionError(f"the Neon stream stopped: {type(self._failure).__name__}: {self._failure}")


def _read_packets(path: Path):
    """The capture's scene packets, in order, as (timestamp, bytes)."""
    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        timestamp, length = PACKET_HEADER.unpack_from(data, offset)
        offset += PACKET_HEADER.size
        yield timestamp, data[offset : offset + length]
        offset += length


def _read_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _discover(config: NeonConfig) -> str:
    # Optional dependency. Only present with the glasses extra.
    from pupil_labs.realtime_api.simple import discover_one_device

    log.info("discovering a Neon, up to %.0f s", config.discovery_timeout_seconds)
    found = discover_one_device(max_search_duration_seconds=config.discovery_timeout_seconds)
    if found is None:
        # Discovery uses mDNS, which university networks routinely block between subnets.
        raise ConnectionError(
            f"no Neon found within {config.discovery_timeout_seconds:.0f} s. "
            "If the network blocks mDNS, read the address off the Companion app's streaming screen and pass --neon-address"
        )
    address = found.address
    found.close()
    log.info("discovered a Neon at %s", address)
    return address


def _measure_time_offset(address: str, port: int, number_of_measurements: int) -> StreamTimeEcho | None:
    # Optional dependency. Only present with the glasses extra.
    from pupil_labs.realtime_api import Device
    from pupil_labs.realtime_api.time_echo import TimeOffsetEstimator

    async def measure() -> StreamTimeEcho | None:
        async with Device(address, port) as device:
            status = await device.get_status()
        if status.phone.time_echo_port is None:
            return None
        estimates = await TimeOffsetEstimator(status.phone.ip, status.phone.time_echo_port).estimate(number_of_measurements)
        if estimates is None:
            return None
        return StreamTimeEcho(StreamEstimate(estimates.time_offset_ms.median), StreamEstimate(estimates.roundtrip_duration_ms.median))

    return asyncio.run(measure())
