"""
Records exactly what a Neon sends over the network, undecoded, so it can be played back later.

Run it while the glasses are worn:

    python examples/capture_neon_stream.py --neon-address 145.137.152.197 --seconds 180 captures/walk_1

The scene video is saved as the compressed packets the glasses send, with their timestamps, so a
playback goes through the same decoder at the same cost as the live stream. Nothing is decoded
while recording, so the capture cannot fall behind the way a decoder can.

What the directory holds:

    scene_packets.bin   one record per packet: float64 timestamp, uint32 length, the bytes
    gaze.jsonl          {"t": capture seconds on the Neon clock, "x": px, "y": px}
    imu.jsonl           {"t": ..., "w": ..., "x": ..., "y": ..., "z": ...}
    meta.json           encoding, SPS and PPS in base64, calibration, clock offset, address, length

All stamps are the Neon's own clock. meta.json carries the Time Echo offset measured at the start,
laptop minus Neon, the same one the live source adds.

meta.json is written as soon as the stream description arrives and again at the end with the final
counts, so a capture stopped with Ctrl+C can still be played back.
"""

# Standard library imports
import argparse
import asyncio
import base64
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.sources.neon_device import apply_opencv_pyav_import_workaround

# The format the replay reads. Defined once, where it is read, so the two cannot drift apart.
from nav.sources.neon_stream import (
    GAZE_FILENAME,
    IMU_FILENAME,
    META_FILENAME,
    PACKET_HEADER,
    SCENE_PACKETS_FILENAME,
)

PROGRESS_INTERVAL_SECONDS = 10.0


async def capture(address: str, port: int, seconds: float, out_dir: Path) -> None:
    """
    Record the scene packets, gaze and IMU for `seconds`, then stop every stream.

    A stream that goes quiet is stopped at the deadline too, rather than holding the capture open.
    """
    # Imported here rather than at the top, because the client imports PyAV and main applies the
    # OpenCV and PyAV workaround first.
    from pupil_labs.realtime_api import Device
    from pupil_labs.realtime_api.streaming.base import RTSPRawStreamer, SDPDataNotAvailableError
    from pupil_labs.realtime_api.streaming.gaze import RTSPGazeStreamer
    from pupil_labs.realtime_api.streaming.imu import RTSPImuStreamer
    from pupil_labs.realtime_api.streaming.video import RTSPVideoFrameStreamer
    from pupil_labs.realtime_api.time_echo import TimeOffsetEstimator

    async with Device(address, port) as device:
        status = await device.get_status()
        world, gaze, imu = status.direct_world_sensor(), status.direct_gaze_sensor(), status.direct_imu_sensor()
        if world is None or not world.connected:
            raise SystemExit("the scene camera is not streaming. Is the Companion app open with the glasses plugged in?")
        calibration = await device.get_calibration()
        offset = None
        if status.phone.time_echo_port is not None:
            estimates = await TimeOffsetEstimator(status.phone.ip, status.phone.time_echo_port).estimate()
            if estimates is not None:
                offset = {"median_ms": estimates.time_offset_ms.median, "round_trip_median_ms": estimates.roundtrip_duration_ms.median}

    out_dir.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + seconds
    counts = {"scene": 0, "gaze": 0, "imu": 0}
    meta = {
        "address": address,
        "seconds": seconds,
        "counts": counts,
        "time_offset": offset,
        "scene_camera_matrix": np.asarray(calibration.scene_camera_matrix).tolist(),
        "scene_distortion_coefficients": np.asarray(calibration.scene_distortion_coefficients).tolist(),
    }

    def write_meta() -> None:
        (out_dir / META_FILENAME).write_bytes(json.dumps(meta, indent=2).encode("utf-8"))

    async def record_scene() -> None:
        async with RTSPVideoFrameStreamer(world.url) as streamer:
            with (out_dir / SCENE_PACKETS_FILENAME).open("wb") as packets:
                # The raw packets, not decoded frames. RTSPRawStreamer.receive is the undecoded stream.
                async for data in RTSPRawStreamer.receive(streamer):
                    if "sprop" not in meta:
                        try:
                            meta["encoding"] = streamer.encoding
                            meta["sprop"] = [base64.b64encode(bytes(param)).decode("ascii") for param in streamer.sprop_parameter_set_payloads]
                        except SDPDataNotAvailableError as not_yet:
                            # The stream description arrives with the first packets. Retried on the next.
                            meta.pop("encoding", None)
                            print(f"waiting for the stream description ({not_yet})", flush=True)
                        else:
                            # Everything a replay needs is known now. Written at once, so a capture
                            # stopped before the deadline can still be played back.
                            write_meta()
                    raw = bytes(data.raw)
                    # One write per packet, so an interrupt cannot leave a header without its bytes.
                    packets.write(PACKET_HEADER.pack(data.timestamp_unix_seconds, len(raw)) + raw)
                    counts["scene"] += 1

    async def record_lines(
        sensor: object | None,
        streamer_class: type,
        filename: str,
        key: str,
        to_record: Callable[[object], dict],
    ) -> None:
        if sensor is None:
            return
        async with streamer_class(sensor.url) as streamer:
            with (out_dir / filename).open("w", encoding="utf-8", newline="\n") as lines:
                async for datum in streamer.receive():
                    lines.write(json.dumps(to_record(datum)) + "\n")
                    counts[key] += 1

    def gaze_record(datum: object) -> dict:
        return {"t": datum.timestamp_unix_seconds, "x": float(datum.x), "y": float(datum.y)}

    def imu_record(datum: object) -> dict:
        quaternion = datum.quaternion
        return {"t": datum.timestamp_unix_seconds, "w": quaternion.w, "x": quaternion.x, "y": quaternion.y, "z": quaternion.z}

    async def progress() -> None:
        while time.monotonic() < deadline:
            await asyncio.sleep(PROGRESS_INTERVAL_SECONDS)
            print(f"{deadline - time.monotonic():5.0f} s left: {counts}", flush=True)

    print(f"recording {seconds:.0f} s to {out_dir}", flush=True)
    try:
        await asyncio.wait_for(
            asyncio.gather(
                record_scene(),
                record_lines(gaze, RTSPGazeStreamer, GAZE_FILENAME, "gaze", gaze_record),
                record_lines(imu, RTSPImuStreamer, IMU_FILENAME, "imu", imu_record),
                progress(),
            ),
            timeout=seconds,
        )
    except TimeoutError:
        # The normal end. Each stream is cancelled where it waits for its next sample, so no record
        # is cut in half.
        pass
    finally:
        # Again with the final counts, also after Ctrl+C. Without a stream description there is
        # nothing a replay could decode, so no meta.json either.
        if "sprop" in meta:
            write_meta()
    print(f"done: {counts}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a Neon's raw stream for playback.")
    parser.add_argument("out_dir", help="a new directory to record into")
    parser.add_argument("--neon-address", required=True)
    parser.add_argument("--neon-port", type=int, default=8080)
    parser.add_argument("--seconds", type=float, default=180.0)
    arguments = parser.parse_args(argv)

    apply_opencv_pyav_import_workaround()
    asyncio.run(capture(arguments.neon_address, arguments.neon_port, arguments.seconds, Path(arguments.out_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
