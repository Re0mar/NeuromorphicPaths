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
"""

# Standard library imports
import argparse
import asyncio
import base64
import json
import struct
import sys
import time
from pathlib import Path

# Third party imports
import numpy as np

# Local package imports
from nav.sources.neon_device import apply_opencv_pyav_import_workaround

PACKET_HEADER = struct.Struct("<dI")  # timestamp in seconds, payload length in bytes


async def capture(address: str, port: int, seconds: float, out_dir: Path) -> None:
    # Optional dependency. Only present with the glasses extra.
    from pupil_labs.realtime_api import Device
    from pupil_labs.realtime_api.streaming.base import RTSPRawStreamer
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
    scene_meta: dict = {}

    async def record_scene() -> None:
        async with RTSPVideoFrameStreamer(world.url) as streamer:
            with (out_dir / "scene_packets.bin").open("wb") as packets:
                # The raw packets, not decoded frames. RTSPRawStreamer.receive is the undecoded stream.
                async for data in RTSPRawStreamer.receive(streamer):
                    if not scene_meta:
                        try:
                            scene_meta["encoding"] = streamer.encoding
                            scene_meta["sprop"] = [base64.b64encode(bytes(param)).decode("ascii") for param in streamer.sprop_parameter_set_payloads]
                        except Exception as not_yet:  # noqa: BLE001, the SDP arrives with the first packets, retried next packet
                            scene_meta.clear()
                            print(f"waiting for the stream description ({type(not_yet).__name__})", flush=True)
                    raw = bytes(data.raw)
                    packets.write(PACKET_HEADER.pack(data.timestamp_unix_seconds, len(raw)))
                    packets.write(raw)
                    counts["scene"] += 1
                    if time.monotonic() >= deadline:
                        return

    async def record_lines(sensor, streamer_class, filename: str, key: str, to_record) -> None:
        if sensor is None:
            return
        async with streamer_class(sensor.url) as streamer:
            with (out_dir / filename).open("w", encoding="utf-8", newline="\n") as lines:
                async for datum in streamer.receive():
                    lines.write(json.dumps(to_record(datum)) + "\n")
                    counts[key] += 1
                    if time.monotonic() >= deadline:
                        return

    def gaze_record(datum) -> dict:
        return {"t": datum.timestamp_unix_seconds, "x": float(datum.x), "y": float(datum.y)}

    def imu_record(datum) -> dict:
        quaternion = datum.quaternion
        return {"t": datum.timestamp_unix_seconds, "w": quaternion.w, "x": quaternion.x, "y": quaternion.y, "z": quaternion.z}

    async def progress() -> None:
        while time.monotonic() < deadline:
            await asyncio.sleep(10.0)
            print(f"{deadline - time.monotonic():5.0f} s left: {counts}", flush=True)

    print(f"recording {seconds:.0f} s to {out_dir}", flush=True)
    await asyncio.gather(
        record_scene(),
        record_lines(gaze, RTSPGazeStreamer, "gaze.jsonl", "gaze", gaze_record),
        record_lines(imu, RTSPImuStreamer, "imu.jsonl", "imu", imu_record),
        progress(),
    )

    meta = {
        "address": address,
        "seconds": seconds,
        "counts": counts,
        "time_offset": offset,
        "scene_camera_matrix": np.asarray(calibration.scene_camera_matrix).tolist(),
        "scene_distortion_coefficients": np.asarray(calibration.scene_distortion_coefficients).tolist(),
        **scene_meta,
    }
    (out_dir / "meta.json").write_bytes(json.dumps(meta, indent=2).encode("utf-8"))
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
