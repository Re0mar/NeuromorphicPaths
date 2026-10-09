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

The same packets also feed an assembler on the decode thread, which gathers each frame's packets
into one access unit for a display that decodes on its own, such as the web page's browser. A
listener subscribed through `subscribe_video` gets the stream's description and then every unit.
Without a listener the units are dropped as they finish.
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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.pose.imu_orientation import is_usable_orientation, orientation_at
from nav.sources.config import NeonConfig
from nav.sources.scene_video import AccessUnit, AccessUnitAssembler, SceneVideoListener

log = logging.getLogger(__name__)

# The capture format, shared with examples/capture_neon_stream.py.
PACKET_HEADER = struct.Struct("<dI")  # timestamp in seconds, payload length in bytes
SCENE_PACKETS_FILENAME = "scene_packets.bin"
GAZE_FILENAME = "gaze.jsonl"
IMU_FILENAME = "imu.jsonl"
META_FILENAME = "meta.json"
# The keys of one gaze or IMU line, stamp first, in the order the replay hands them on.
GAZE_FIELDS = ("t", "x", "y")
IMU_FIELDS = ("t", "w", "x", "y", "z")

# How far a gaze sample may be from a frame and still count as that frame's gaze. A frame is 33 ms.
GAZE_MATCH_TOLERANCE_SECONDS = 0.05
GAZE_HISTORY = 400  # About two seconds of gaze at 200 Hz.
# Three seconds of usable IMU readings at about 110 Hz. A frame can reach the pipeline 1.6 s after it
# was captured (the worst seen on the 2026-10-05 walk), and its orientation has to still be in here.
IMU_HISTORY = 330
# Decoder threads. Frame threading would add a frame of delay per thread. Slice threading costs
# nothing, and helps whenever the phone's encoder splits a frame into slices.
DECODER_THREAD_TYPE = "SLICE"
# The decode thread checks for a stop this often when no packets come.
DECODE_POLL_SECONDS = 0.25
# Half a second of scene packets at about 600 a second. More than that waiting means the decoder is
# falling behind, which is the delay this module exists to prevent, so it is logged.
DECODE_BACKLOG_WARNING_PACKETS = 300
BACKLOG_WARNING_INTERVAL_SECONDS = 10.0
# A wifi drop can damage a run of packets in a row. One line per interval says so without a flood.
CORRUPT_PACKET_WARNING_INTERVAL_SECONDS = 10.0
# How long a replay sleeps at a time while waiting for the next packet to be due.
REPLAY_SLEEP_SLICE_SECONDS = 0.1
# Generous, because it only fires when the receiver never got going at all.
START_WAIT_GRACE_SECONDS = 30.0


class ReplayEventKind(Enum):
    """What one entry in a played-back capture is."""

    SCENE = "scene"
    GAZE = "gaze"
    IMU = "imu"


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
class StreamQuaternion:
    w: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class StreamMatched:
    frame: StreamFrame
    gaze: StreamGaze | None
    # The usable IMU reading nearest the frame's capture stamp, or None when none was near enough.
    orientation: StreamQuaternion | None


@dataclass(frozen=True)
class StreamImuStatus:
    """What the IMU has sent since the stream started, for a check to judge it by."""

    readings: int
    empty_readings: int
    # Empty readings whose timestamp was zero as well. On 2026-10-05 every empty one was like
    # this, a packet that decoded to nothing at all rather than a stamped one with no rotation.
    unstamped_empty_readings: int
    frames_without_orientation: int


def client_failure_types() -> tuple[type[BaseException], ...]:
    """
    The failures the Pupil Labs client raises when the glasses or the network let it down.

    The client's own DeviceError, the HTTP library's ClientError, and the RTSP library's RTSPError.
    A dropped HTTP connection is a ClientError and not an OSError, so without these it would read
    as a failure nobody predicted.

    :return: The types, or none at all without the glasses extra.
    :rtype: tuple[type[BaseException], ...]
    """
    try:
        # Optional dependency. Absent in any environment installed without the glasses extra.
        from aiortsp.rtsp.errors import RTSPError
        from pupil_labs.realtime_api import device as client_device
    except ImportError:
        return ()
    # Taken from the client's module rather than imported. The HTTP library is the web sink's
    # import, and a guard test keeps it to that one file.
    return (client_device.DeviceError, client_device.aiohttp.ClientError, RTSPError)


class SceneDecoder:
    """H.264 packets in, the newest decoded frame kept. Nothing converted until asked for."""

    def __init__(self, sprop_parameter_sets: list[bytes]) -> None:
        """
        :param sprop_parameter_sets: The stream's SPS and PPS as the client's
            `sprop_parameter_set_payloads` gives them, start code included. Live they come straight
            from the client, and a capture stores the same bytes.
        """
        # Optional dependency. PyAV comes with the glasses extra, through the Pupil Labs client.
        import av

        # Same helper the client's own decoder uses. The packets arrive as RTP payloads.
        from pupil_labs.realtime_api.streaming.nal_unit import extract_payload_from_nal_unit

        self._extract = extract_payload_from_nal_unit
        self._codec = av.CodecContext.create("h264", "r")
        self._codec.thread_type = DECODER_THREAD_TYPE
        for parameter_set in sprop_parameter_sets:
            self._codec.parse(parameter_set)
        self._frame_timestamp: float | None = None

    def feed(self, raw: bytes, timestamp_unix_seconds: float) -> tuple[object, float] | None:
        """
        One packet in. Returns the newest decoded frame and its stamp, if this packet completed one.

        A packet that starts a new frame flushes the previous one, which carries the previous
        packet's stamp. That is how the client does it, and why the stamp lags by one packet here.

        :raises ValueError: For a packet the NAL helper or the decoder cannot read.
        :raises struct.error: For an empty packet.
        """
        newest = self._decode_parsed(self._codec.parse(self._extract(raw)))
        self._frame_timestamp = timestamp_unix_seconds
        return newest

    def flush(self) -> tuple[object, float] | None:
        """
        The frame still held back after the last packet, if there is one.

        The parser only lets a frame go when the next one starts, so without this the last frame
        of a stream never comes out. It carries the last packet's stamp.
        """
        newest = self._decode_parsed(self._codec.parse(None))
        for frame in self._codec.decode(None):
            if self._frame_timestamp is not None:
                newest = (frame, self._frame_timestamp)
        return newest

    def _decode_parsed(self, packets: list) -> tuple[object, float] | None:
        newest = None
        for packet in packets:
            for frame in self._codec.decode(packet):
                if self._frame_timestamp is not None:
                    newest = (frame, self._frame_timestamp)
        return newest


class NeonStreamDevice:
    """Receives on a background thread, and answers the simple Device's calls from what it kept."""

    def __init__(
        self,
        config: NeonConfig,
        decoder_factory: Callable[[list[bytes]], "SceneDecoder"] | None = None,
    ) -> None:
        """
        :param config: Where the glasses are, or which capture to play back.
        :param decoder_factory: Builds the decoder from the stream's parameter sets. None means the
            real H.264 decoder. Tests pass one that needs no PyAV.
        """
        self._config = config
        self._decoder_factory = decoder_factory if decoder_factory is not None else SceneDecoder
        self._condition = threading.Condition()
        self._newest_frame: tuple[object, float] | None = None
        self._frames_decoded = 0
        self._frames_served = 0
        self._gaze: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=GAZE_HISTORY)
        # Usable readings only, (stamp, w, x, y, z), oldest first. Empty ones are counted, never kept.
        self._imu: collections.deque[tuple[float, float, float, float, float]] = collections.deque(maxlen=IMU_HISTORY)
        self._imu_readings = 0
        self._imu_empty_readings = 0
        self._imu_unstamped_empty_readings = 0
        self._frames_without_orientation = 0
        self._failure: BaseException | None = None
        self._replay_finished = False
        self._stopping = threading.Event()
        self._ready = threading.Event()
        self._calibration: StreamCalibration | None = None
        self._time_offset: StreamTimeEcho | None = None
        self._replay_shift_seconds = 0.0
        # The address actually connected to, which is the discovered one when none was given.
        self._address: str | None = config.address
        self._thread = threading.Thread(target=self._run, name="neon-receive", daemon=True)
        self._packets: queue.Queue = queue.Queue()
        self._last_backlog_warning = 0.0
        self._corrupt_packets = 0
        self._last_corrupt_warning: float | None = None
        # The tee for a display that decodes on its own. Both under _condition, since the decode
        # thread reads them and the pipeline's thread sets them.
        self._video_listener: SceneVideoListener | None = None
        self._assembler: AccessUnitAssembler | None = None
        self._decode_thread = threading.Thread(target=self._decode, name="neon-decode", daemon=True)

    def start(self) -> None:
        """Connect, read the calibration and the clock offset, and start receiving."""
        self._decode_thread.start()
        self._thread.start()
        if not self._ready.wait(self._config.discovery_timeout_seconds + START_WAIT_GRACE_SECONDS):
            raise ConnectionError("the Neon stream did not start within the discovery timeout")
        if self._failure is not None:
            raise self._failure

    # The simple Device's calls. Shapes match the client's, field for field.

    def get_calibration(self) -> StreamCalibration:
        if self._calibration is None:
            raise ValueError("no calibration was read when the stream started")
        return self._calibration

    def estimate_time_offset(self, number_of_measurements: int = 100) -> StreamTimeEcho | None:
        """
        Laptop clock minus Neon clock. On a replay, the offset stored with the capture, or None.

        :return: The medians, or None when the Companion app cannot answer, when the measurement
            ran past `time_echo_timeout_seconds`, or when the capture was recorded without one.
        """
        if self._config.replay_dir is not None:
            return self._time_offset
        return _measure_time_offset(
            self._address,
            self._config.port,
            number_of_measurements,
            self._config.time_echo_timeout_seconds,
        )

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
            # From when the frame was captured, not from now. The frame handed over is the newest
            # decoded one, which can be 1.6 s old by the time it's asked for, and the head has moved.
            orientation = self._orientation_at(timestamp)
            if orientation is None:
                self._frames_without_orientation += 1
        # Converted here, once, for the one frame that is used.
        pixels = frame.to_ndarray(format="bgr24")
        return StreamMatched(StreamFrame(pixels, timestamp), gaze, orientation)

    def imu_status(self) -> StreamImuStatus:
        """What the IMU has sent so far: readings, how many were empty, and frames left without one."""
        with self._condition:
            self._raise_if_failed()
            return StreamImuStatus(
                readings=self._imu_readings,
                empty_readings=self._imu_empty_readings,
                unstamped_empty_readings=self._imu_unstamped_empty_readings,
                frames_without_orientation=self._frames_without_orientation,
            )

    def subscribe_video(self, listener: SceneVideoListener) -> None:
        """
        Hand every finished access unit to the listener, after the stream's description.

        A listener that arrives before the stream is described gets the description when the
        first packets bring it. One that arrives later gets it at once, and its first unit is the
        next one to finish, so it starts clean at the next keyframe.
        """
        with self._condition:
            self._video_listener = listener
            assembler = self._assembler
        if assembler is not None:
            listener.describe(assembler.description)

    def _build_assembler(self, parameter_sets: list[bytes]) -> None:
        """
        Start assembling units from the stream's parameter sets, and describe the stream to a listener already waiting.

        A stream whose parameter sets hold no usable sequence parameter set gets no tee: the
        decoder may still make sense of it, and the depth frames matter more than the page's
        video, so the run goes on and the page is told nothing rather than something wrong.
        """
        try:
            assembler = AccessUnitAssembler(parameter_sets)
        except ValueError as unusable:
            log.warning("no scene video for a display that decodes on its own (caught ValueError, expected): %s", unusable)
            return
        with self._condition:
            self._assembler = assembler
            listener = self._video_listener
        if listener is not None:
            listener.describe(assembler.description)

    def close(self) -> None:
        self._stopping.set()
        # A device that was never started has no threads to wait for, and joining one raises.
        if self._thread.ident is not None:
            self._thread.join(5.0)
        if self._decode_thread.ident is not None:
            self._decode_thread.join(5.0)

    # The receiving side, all on the background thread.

    def _run(self) -> None:
        try:
            asyncio.run(self._receive())
        except (*client_failure_types(), OSError, ValueError) as failure:
            # The glasses or the network went away, or the capture is malformed. Handed to the
            # caller's thread and raised there, which ends the run with this message.
            self._fail(failure)
            log.error("the Neon stream stopped (caught %s, expected): %s", type(failure).__name__, failure)
        except Exception as unexpected:  # noqa: BLE001, handed to the caller's thread and raised there
            self._fail(unexpected)
            log.error("UNEXPECTED %s in the Neon stream, may need a handler", type(unexpected).__name__, exc_info=True)
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
        from pupil_labs.realtime_api.streaming.base import RTSPRawStreamer, SDPDataNotAvailableError
        from pupil_labs.realtime_api.streaming.gaze import RTSPGazeStreamer
        from pupil_labs.realtime_api.streaming.imu import RTSPImuStreamer
        from pupil_labs.realtime_api.streaming.video import RTSPVideoFrameStreamer

        # The client's discovery helper runs its own event loop, which cannot start inside this
        # one. On a worker thread it has none to collide with. Discovery had never run live before
        # 2026-10-08, every session having passed the address, and the crash was waiting here.
        address = self._config.address or await asyncio.to_thread(_discover, self._config)
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
        stream_failures = (*client_failure_types(), OSError, ValueError)

        async def scene() -> None:
            async with RTSPVideoFrameStreamer(world.url) as streamer:
                decoder = None
                async for data in RTSPRawStreamer.receive(streamer):
                    if decoder is None:
                        try:
                            parameter_sets = [bytes(param) for param in streamer.sprop_parameter_set_payloads]
                        except SDPDataNotAvailableError as not_yet:
                            # The stream description arrives with the first packets. Retried on the next.
                            log.debug("waiting for the stream description (%s)", not_yet)
                            continue
                        decoder = self._decoder_factory(parameter_sets)
                        self._build_assembler(parameter_sets)
                    self._on_packet(decoder, bytes(data.raw), data.timestamp_unix_seconds)
                    if self._stopping.is_set():
                        return

        async def gaze_stream() -> None:
            if gaze is None:
                return
            try:
                async with RTSPGazeStreamer(gaze.url) as streamer:
                    async for datum in streamer.receive():
                        self._on_gaze(datum.timestamp_unix_seconds, float(datum.x), float(datum.y))
                        if self._stopping.is_set():
                            return
            except stream_failures as failure:
                # Gaze is optional. A walk without it still plans, so frames go on without gaze
                # rather than the run ending over the pointer.
                log.warning(
                    "the Neon's gaze stream stopped (caught %s, expected), frames carry no gaze from here: %s",
                    type(failure).__name__,
                    failure,
                )

        async def imu_stream() -> None:
            if imu is None:
                return
            try:
                async with RTSPImuStreamer(imu.url) as streamer:
                    async for datum in streamer.receive():
                        quaternion = datum.quaternion
                        self._on_imu(datum.timestamp_unix_seconds, quaternion.w, quaternion.x, quaternion.y, quaternion.z)
                        if self._stopping.is_set():
                            return
            except stream_failures as failure:
                # Not carried on without. The source would keep handing out the last orientation,
                # and the floor would be fitted to where the head was, not where it is.
                raise ConnectionError(f"the Neon's IMU stream stopped (caught {type(failure).__name__}): {failure}") from failure

        await asyncio.gather(scene(), gaze_stream(), imu_stream())

    async def _receive_replay(self, capture: Path) -> None:
        calibration, self._time_offset, parameter_sets = _read_meta(capture)
        self._calibration = calibration
        decoder = self._decoder_factory(parameter_sets)
        self._build_assembler(parameter_sets)

        packets = list(read_capture_packets(capture / SCENE_PACKETS_FILENAME))
        gaze = read_capture_samples(capture / GAZE_FILENAME, GAZE_FIELDS)
        imu = read_capture_samples(capture / IMU_FILENAME, IMU_FIELDS)
        if not packets:
            raise ValueError(f"{capture} holds no scene packets")
        first_stamp = packets[0][0]
        started = time.monotonic()
        # Moves the recording's Neon stamps to now. With the recorded offset added, a packet's
        # capture time on the laptop clock is the moment it is fed, so lag measures this laptop.
        offset_ms = 0.0 if self._time_offset is None else self._time_offset.time_offset_ms.median
        self._replay_shift_seconds = (time.time() - offset_ms / 1000.0) - first_stamp
        # The shift in full, so a replay's frame log can be matched back to the capture's own stamps.
        # Fitting it afterwards can't work: frames are 33.3 ms apart to within 11 us, so a fit one
        # frame off looks as good as the right one.
        log.info(
            "replaying %d scene packets, %.0f s, from %s, stamps shifted by %r s",
            len(packets),
            packets[-1][0] - first_stamp,
            capture.name,
            self._replay_shift_seconds,
        )
        self._ready.set()

        events = sorted(
            [(stamp, ReplayEventKind.SCENE, payload) for stamp, payload in packets]
            + [(sample[0], ReplayEventKind.GAZE, sample) for sample in gaze]
            + [(sample[0], ReplayEventKind.IMU, sample) for sample in imu],
            key=lambda event: event[0],
        )
        for stamp, kind, payload in events:
            if self._stopping.is_set():
                return
            # In slices, so a close lands within one rather than after the gap to the next packet.
            wait = (stamp - first_stamp) - (time.monotonic() - started)
            while wait > 0 and not self._stopping.is_set():
                await asyncio.sleep(min(wait, REPLAY_SLEEP_SLICE_SECONDS))
                wait = (stamp - first_stamp) - (time.monotonic() - started)
            if self._stopping.is_set():
                return
            shifted = stamp + self._replay_shift_seconds
            match kind:
                case ReplayEventKind.SCENE:
                    self._on_packet(decoder, payload, shifted)
                case ReplayEventKind.GAZE:
                    _, x, y = payload
                    self._on_gaze(shifted, x, y)
                case ReplayEventKind.IMU:
                    _, w, x, y, z = payload
                    # A zero stamp is a packet that carried none, as on 2026-10-05, not a moment in
                    # time. Shifted, it would look stamped, and a replay would hide that.
                    self._on_imu(shifted if stamp != 0.0 else 0.0, w, x, y, z)
        # The parser holds the last frame until another one starts, and none will.
        self._packets.put((decoder, None, 0.0))
        # Finished means decoded too, or the last frames would be lost to the end-of-replay check.
        # Polled rather than joined. A join on a helper thread waited for good once a close had
        # stopped the decoder with packets still queued, and Python waits for that thread on exit.
        while self._packets.unfinished_tasks and not self._stopping.is_set() and self._failure is None:
            await asyncio.sleep(REPLAY_SLEEP_SLICE_SECONDS)
        if self._stopping.is_set() or self._failure is not None:
            return
        with self._condition:
            frames_decoded = self._frames_decoded
        if frames_decoded == 0:
            # Reported as a failure rather than a normal end, because a run of zero frames that
            # exits cleanly looks exactly like a short walk.
            raise ValueError(f"the capture in {capture} decoded no frames from its {len(packets)} scene packets")
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
                # No packet means the stream has ended and the decoder lets go of what it holds.
                decoded = decoder.flush() if raw is None else decoder.feed(raw, timestamp)
            except (ValueError, struct.error) as corrupt:
                # One damaged packet. H.264 picks up again at the next keyframe, so the packet is
                # dropped and decoding goes on.
                self._report_corrupt_packet(corrupt)
                continue
            except Exception as unexpected:  # noqa: BLE001, handed to the caller's thread and raised there
                # Not a damaged packet anyone predicted, so not skipped like one. The next receive
                # raises it, rather than the pipeline waiting on frames that will never come.
                log.error("UNEXPECTED %s decoding a scene packet, may need a handler", type(unexpected).__name__, exc_info=True)
                self._fail(unexpected)
                return
            finally:
                self._packets.task_done()
            self._tee_packet(raw, timestamp)
            if decoded is None:
                continue
            with self._condition:
                self._newest_frame = decoded
                self._frames_decoded += 1
                self._condition.notify_all()

    def _tee_packet(self, raw: bytes | None, timestamp: float) -> None:
        """
        The same packet to the assembler, after the decoder has had it, and a finished unit to the listener.

        Its own try, apart from the decoder's, so a packet the assembler refuses costs the display
        one packet and never the decoded frame. No packet is the end of the stream, and flushes.
        """
        with self._condition:
            assembler = self._assembler
            listener = self._video_listener
        if assembler is None:
            return
        try:
            unit = assembler.flush() if raw is None else assembler.feed(raw, timestamp)
        except ValueError as corrupt:
            # The same damaged packet the decoder skipped, or one only the assembler minds.
            self._report_corrupt_packet(corrupt)
            return
        if unit is not None and listener is not None:
            listener.offer(unit)

    def _report_corrupt_packet(self, corrupt: Exception) -> None:
        self._corrupt_packets += 1
        now = time.monotonic()
        if self._last_corrupt_warning is None or now - self._last_corrupt_warning > CORRUPT_PACKET_WARNING_INTERVAL_SECONDS:
            log.warning(
                "skipped a scene packet the decoder could not read (caught %s, expected), %d so far: %s",
                type(corrupt).__name__,
                self._corrupt_packets,
                corrupt,
            )
            self._last_corrupt_warning = now

    def _fail(self, failure: BaseException) -> None:
        with self._condition:
            # The first failure is the cause. Anything after it is usually its echo.
            if self._failure is None:
                self._failure = failure
            self._condition.notify_all()

    def _on_gaze(self, timestamp: float, x: float, y: float) -> None:
        with self._condition:
            self._gaze.append((timestamp, x, y))

    def _on_imu(self, timestamp: float, w: float, x: float, y: float, z: float) -> None:
        usable = is_usable_orientation(np.array([w, x, y, z], dtype=np.float64))
        with self._condition:
            self._imu_readings += 1
            if not usable:
                # The glasses sent only zeros for minutes on 2026-10-05. A zero is no orientation, so
                # it never stands in for one. The count is how a check finds out.
                self._imu_empty_readings += 1
                if timestamp == 0.0:
                    self._imu_unstamped_empty_readings += 1
                return
            self._imu.append((timestamp, w, x, y, z))

    def _orientation_at(self, timestamp: float) -> StreamQuaternion | None:
        """The usable IMU reading nearest a frame's stamp, by the one rule every route uses. Caller holds the lock."""
        if not self._imu:
            return None
        samples = np.array(self._imu, dtype=np.float64)
        # Sorted here rather than trusted, since a reading that arrives late over the network would
        # otherwise sit out of order and the nearest-stamp search would read past it.
        samples = samples[np.argsort(samples[:, 0], kind="stable")]
        orientation = orientation_at(samples[:, 0], samples[:, 1:], timestamp)
        if orientation is None:
            return None
        return StreamQuaternion(*(float(component) for component in orientation))

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


def _read_meta(capture: Path) -> tuple[StreamCalibration, StreamTimeEcho | None, list[bytes]]:
    """
    The capture's calibration, its recorded clock offset and the stream's parameter sets.

    :raises ValueError: For a meta.json that is not the format the capture script writes.
    """
    try:
        meta = json.loads((capture / META_FILENAME).read_text(encoding="utf-8"))
        calibration = StreamCalibration(
            np.asarray(meta["scene_camera_matrix"], dtype=np.float64),
            np.asarray(meta["scene_distortion_coefficients"], dtype=np.float64),
        )
        parameter_sets = [base64.b64decode(param) for param in meta["sprop"]]
        # Null when the Companion app could not answer Time Echo while recording.
        offset = meta.get("time_offset")
        time_offset = None
        if offset is not None:
            time_offset = StreamTimeEcho(StreamEstimate(float(offset["median_ms"])), StreamEstimate(float(offset["round_trip_median_ms"])))
    except KeyError as missing:
        raise ValueError(f"capture {capture} is malformed: {META_FILENAME} has no {missing}") from missing
    except (TypeError, ValueError) as malformed:
        # ValueError covers broken JSON and broken base64, which both derive from it.
        raise ValueError(f"capture {capture} is malformed: {META_FILENAME} ({type(malformed).__name__}: {malformed})") from malformed
    return calibration, time_offset, parameter_sets


def read_capture_packets(path: Path) -> Iterator[tuple[float, bytes]]:
    """
    The capture's scene packets, in order, as (timestamp, bytes).

    :raises ValueError: For a file that ends partway through a header or a payload, naming where.
    """
    data = path.read_bytes()
    offset = 0
    while offset < len(data):
        if offset + PACKET_HEADER.size > len(data):
            raise ValueError(f"capture {path.parent} is malformed: {path.name} ends partway through a packet header at byte {offset}")
        timestamp, length = PACKET_HEADER.unpack_from(data, offset)
        payload_start = offset + PACKET_HEADER.size
        if payload_start + length > len(data):
            raise ValueError(
                f"capture {path.parent} is malformed: {path.name} has a packet at byte {offset} "
                f"of {length} bytes, and only {len(data) - payload_start} remain"
            )
        yield timestamp, data[payload_start : payload_start + length]
        offset = payload_start + length


def read_capture_samples(path: Path, field_names: tuple[str, ...]) -> list[tuple[float, ...]]:
    """
    Each line of a gaze or IMU file as its fields in the order named, stamp first.

    A missing file is a capture recorded without that stream, so it reads as no samples.

    :raises ValueError: For a line that is not JSON or lacks a field, naming the file and the line.
    """
    if not path.is_file():
        return []
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            samples.append(tuple(float(record[name]) for name in field_names))
        except (KeyError, TypeError, ValueError) as malformed:
            raise ValueError(
                f"capture {path.parent} is malformed: {path.name} line {line_number} ({type(malformed).__name__}: {malformed})"
            ) from malformed
    return samples


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


def _measure_time_offset(address: str, port: int, number_of_measurements: int, timeout_seconds: float) -> StreamTimeEcho | None:
    """
    Laptop clock minus Neon clock by the client's Time Echo protocol, given up after a bound.

    The client takes no timeout of its own, and a phone that has gone away would hold the device
    process until it did. The parent waits for its answer, so this bound is the one that counts.

    :return: The medians, or None when the phone has no Time Echo port or did not answer in time.
    """
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

    try:
        return asyncio.run(asyncio.wait_for(measure(), timeout_seconds))
    except TimeoutError:
        log.warning("Time Echo did not answer within %.1f s, so this clock offset is left unmeasured", timeout_seconds)
        return None
