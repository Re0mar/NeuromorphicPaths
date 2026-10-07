# Depth to laptop, the Pixel app

Written for whoever runs or extends the phone side. The app takes ARCore's depth image, camera
pose and floor plane off a Pixel and sends them to the laptop pipeline in `server/` over TCP, in
the format `server/docs/arcore_wire_format.md` describes. The laptop plans a path and sends it
back over a second connection, and the app draws the camera picture with an arrow over it
pointing where the path says. The walker looks at the phone and nothing else.

## Phones

Nothing in the app is tied to a Pixel. It has run on a Pixel 8 so far, and any phone that meets
these three requirements should work:

- **Android 14 or newer.** That's `minSdk 34`, so an older phone can't install the app at all.
- **ARCore.** The manifest marks it as required, so the Play Store asks to install Google Play
  Services for AR if it's missing. A phone not on Google's ARCore device list can't run the app.
- **ARCore's depth mode.** The app checks `isDepthModeSupported(Config.DepthMode.AUTOMATIC)` when
  it opens the AR session. Without depth it doesn't open the session, and the status line reads
  "this device has no ARCore Depth API". Not every ARCore phone has depth.

## Build

Android Studio's JDK and the Android SDK, both already on the build laptop. From this folder:

```
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ANDROID_HOME="$LOCALAPPDATA/Android/Sdk" ./gradlew.bat :app:assembleDebug
```

The debug APK lands in `app/build/outputs/apk/debug/`. The same toolchain versions as the group's
app on the `restart` branch: Gradle 9.6, AGP 9.4.1, Kotlin 2.2.10, minSdk 34.

For any walk whose timing goes in a report, build the release APK instead. A debuggable build draws
slower, and the arrow's draw time is part of what gets measured. It's signed with the debug key so
it installs like the debug one, and it isn't meant for a store:

```
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ANDROID_HOME="$LOCALAPPDATA/Android/Sdk" ./gradlew.bat :app:assembleRelease
adb install -r app/build/outputs/apk/release/app-release.apk
```

The debug key belongs to the laptop that built the APK. A phone holding the app from another
laptop's build refuses this one with `INSTALL_FAILED_UPDATE_INCOMPATIBLE`. Uninstall the app
first in that case, but **pull the phone's files first**. Uninstalling deletes the app's whole
folder, every timing log and every recorded walk with it:

```
MSYS_NO_PATHCONV=1 adb pull /sdcard/Android/data/com.neuromorphicpaths.pixel/files/ phone_files_backup/
```

Anyone holding a laptop's debug key can sign an update for an app built with it, so this build
is for the team's own phones and nothing else.

## Run

1. Start the laptop listening, from `server/`:
   `python -m nav --source arcore_tcp --arcore-accept-timeout 600 --reconnect --sink phone_app --sink web --floor-max-tilt 50`.
   That serves the arrow to this app and the depth view to a browser at `http://<laptop>:8765` in
   the same run. It listens on three ports, 9000 for depth, 9100 for paths and 8765 for the page.
   Each needs an inbound firewall rule on the laptop, and the rule has to name the Python that
   owns the socket, which with a venv is the base interpreter and not the venv's launcher. See
   `server/README.md`.
2. Install and open the app. It asks for the camera, then for ARCore if the phone lacks it. On
   Android 17 it also asks for local network access. Allow it: a laptop on home or lab Wi-Fi sits
   at a private address such as `192.168.x` or `10.x`, and without the permission the connection
   times out. A refusal shows as a Permission line above the status lines.
3. Type the laptop's address, leave the two ports unless the laptop was started with others,
   and tap Connect. The app opens both connections to that one address.

What the screen shows, top to bottom: the three status lines, then the camera picture with the
arrow over it.

- Line one: the depth connection, frames sent and frames dropped as stale.
- Line two: whether ARCore is tracking, how many frames had depth, and whether a floor is found.
- Line three: the path connection, paths received and refused, and the last refusal's reason.
- The arrow points where the newest path says, positive to the right, with the heading in
  degrees under it. Green normally, red with `ALARM` when something in the walker's way is
  close, grey with `No path yet` before the first path. The age under the heading counts up
  from when the path arrived and says `stale` after a second without a new one, so a frozen
  arrow is visibly frozen.

The phone and the laptop have to reach each other. On a phone hotspot the laptop is usually
`192.168.43.1`, which is the default on screen. On the emulator the laptop is `10.0.2.2`. Over
USB, `adb reverse tcp:9000 tcp:9000` and `adb reverse tcp:9100 tcp:9100` make `127.0.0.1` on the
phone reach the laptop. Started from adb, a debug build connects by itself. A release build
ignores these extras, because the activity is exported and any app on the phone could start it
pointed at an address of its choosing:

```
adb shell am start -n com.neuromorphicpaths.pixel/.MainActivity --es host 127.0.0.1 --ei port 9000 --ei path_port 9100
```

Installing on the emulator from this machine: use `adb push` and `pm install`, not
`adb install`. The streamed install wedges the emulator's package manager.

## Recording a walk without the laptop

For a walk somewhere the phone can't reach the laptop. The phone records the walk on its own, and
later sends the recording to the laptop as if it were happening then. The laptop needs nothing
new: it records the replay with `--record-to` like any live walk.

1. Tap **Record walk** and walk. Nothing needs to be connected. ARCore tracks and the frames go to
   a file in the app's own storage, `walks/walk_<date>_<time>.bin`, the time to the millisecond.
   Tap **Stop recording** at the end. The button reads **Saving** until the last queued frames are
   written, and Record and Replay both wait for it. The line under the buttons counts frames
   written, frames dropped and megabytes. A drop means the phone fell behind writing, and the count
   says how much of the walk is missing. A frame whose values can't go in the file, such as a NaN in
   the pose, is left out and counted as unencodable.
2. Later, where the phone reaches the laptop, make sure no live connection is running: swipe the
   app away, or `adb shell am force-stop com.neuromorphicpaths.pixel`, and don't tap Connect after
   reopening it. A live connection retries in the background, and the laptop records whichever
   connection reaches it first. If that's the live one, the laptop records tonight's live frames,
   and when the replay stops the live connection the laptop takes that as the end of the walk and
   shuts down. Then start the laptop recording into a fresh directory:
   `python -m nav --source arcore_tcp --arcore-accept-timeout 600 --sink phone_app --sink web --floor-max-tilt 50 --record-to frame_logs/<name>`.
3. Type the laptop's address and tap **Replay latest walk**. The app stops its live connection,
   because the laptop takes one depth connection at a time, and sends the newest recording,
   every frame unchanged and in order, at the pace it was recorded. The laptop's log gets the
   original timestamps. After the last frame the phone closes its side and waits up to 30 s for
   the laptop to close back, which the laptop does once it has read every frame. The line then
   says the replay finished and whether the laptop read them all. If it says the laptop didn't
   confirm, check the laptop's frame count before trusting the log. Then stop the laptop with
   Ctrl-C if it hasn't stopped on its own.

A replay never drops a frame and never resumes. If the connection is lost partway, the line says
so with the count sent, and the laptop's log is missing the rest of the walk. Replay again into a
fresh directory.

A recording is the wire stream itself, a 4-byte length before each frame, about 30 KB a frame,
roughly 1 MB a second. Recordings stay on the phone until deleted, and can be copied off with
`adb pull /sdcard/Android/data/com.neuromorphicpaths.pixel/files/walks/`.

## Emulator

The emulator cannot produce depth, so it only proves the plumbing: install, permission, ARCore
session, connection to the laptop at `10.0.2.2`. Three things have to be right for even that.

- Install Google's ARCore build for the emulator (the `_x86_for_emulator` APK from the ARCore SDK
  releases) before the app, by push and `pm install`.
- Start the emulator with `-camera-back environment`. That is emulator 36's name for the virtual
  scene camera. `virtualscene` is not a value, and an unknown value leaves the device with no
  back camera and no error.
- ARCore 1.56's emulator build asks the camera service for camera `0` by name. On emulator
  36.6 with the API 34 and API 36.1 Google APIs x86_64 images the back camera is id `10`, so
  `Session()` throws `FatalException` and no session exists. The app catches that and shows it
  as a disconnected reason. `adb shell dumpsys media.camera` lists the ids. Until Google's
  emulator build and the emulator's camera HAL agree again, the session itself is only testable
  on the phone, and the emulator proves the install and the connection.

## What it sends and why

- Depth as `uint16` millimeters straight from `acquireDepthImage16Bits`, rows repacked to drop
  any stride padding, the top three bits masked because the DEPTH16 format reserves them.
- The camera pose converted from ARCore's y-up, z-back camera axes to the laptop's y-down,
  z-forward ones, into ARCore's world, which has y up. The laptop reads gravity from that, and
  needs to: the depth image is in the sensor's landscape orientation however the phone is held,
  so with the phone in portrait the image's own up points sideways. `has_position` is false
  whenever tracking is not `TRACKING`, because a stale position is worse than none.
- The largest upward-facing plane ARCore tracks, by extent, as the floor, in the camera frame,
  or `null` when ARCore has none yet. Not the lowest: on the first walk ARCore also tracked a
  plane a meter below the real floor, and the lowest-plane rule sent that one. The laptop gates
  whatever it is sent against where a floor can be, and fits its own floor when the plane fails
  or is missing.
- Intrinsics scaled from the GPU camera texture to the depth image's size. The depth image
  covers the texture's 16:9 view, not the 4:3 CPU image, and scaling the wrong one made the
  horizontal focal length a third too long on the first run.
- One message per ARCore frame. The surface draws at the display rate, and a draw that gets the
  frame the previous draw already sent is skipped. The camera picture is still drawn on every
  draw, so the preview refreshes at the display rate while depth goes out at the camera's.

## What it receives

Each planned path arrives on the second connection as a length-prefixed JSON message, the shape
in `server/docs/arcore_wire_format.md` under what the laptop sends back. The app checks every
field before it reads it: every key present, every number a number and finite, the alarm a
boolean, the two arrays the same length and not empty. A message that breaks a rule is dropped,
counted on line three with the rule it broke, and the connection stays up. A zero or oversized
length prefix means the stream is out of step, so the app closes the connection and reconnects,
the same thing the laptop does with a bad prefix on the depth side.

## Timing log

Every time the app starts it writes a timing log for measuring the delay from a depth frame to
the arrow drawn from it. One JSON line per moment: the session (phone model, build type, start
time), then for each ARCore frame when it was handled, when its depth was handed to the socket or
that it never went, when its path came back, and when the arrow was first drawn with that path.
"Handed to the socket" is when the write returns, so time the bytes then wait in the phone's send
buffer counts as network. "Drawn" is when the draw commands are issued, before the screen
shows them. A frame that never went was replaced by a newer one, caught by a failed write, or
still waiting when the connection stopped.
Every time in it is the phone's elapsed-realtime clock in nanoseconds, and every line names its
frame by ARCore's own timestamp, which is what the laptop hands back in each path. The laptop's
timing report joins this file with the laptop's own log on that timestamp. Every line takes that
key from the same seconds the frame travels in, so the lines still match past 47 days of uptime,
where a double can no longer hold the stamp to the nanosecond.

The log goes to the app's own external files folder, one file per session named by its start time:

```
MSYS_NO_PATHCONV=1 adb pull /sdcard/Android/data/com.neuromorphicpaths.pixel/files/timing/ timing_from_phone/
```

`MSYS_NO_PATHCONV=1` stops Git Bash rewriting the device path into a Windows one. Pull with
`adb pull`, never `adb shell cat`, which adds carriage returns. A log stops at 20 MB with a
`truncated` line, and a `lost` line counts any records the writer was too far behind to keep or
that arrived while the app was closing.

The log starts when the app does, not at Connect, and old logs are never deleted. A walk on
2026-10-07 wrote 900 KB in about 3 minutes, 5.1 KB a second, so one session reaches 20 MB after
about an hour. Start the app shortly before a walk rather than leaving it open, and clear old
logs from the folder now and then.

## Tests

`./gradlew.bat :app:testDebugUnitTest` runs the JVM tests: the two messages' own checks on their
shapes, the encoder, the path decoder one rule per test, both connections against loopback
servers, the walk recorder, its file reader and its replayer against a loopback server, the floor
choice, the arrow's arithmetic, the ARCore conversion's pure functions (the quaternion, the
floor plane, the DEPTH16 repacking), and the timing log against an in-memory output and a
counter clock. Three of them are the contract checks across the language boundary:

- The encoder test writes `app/build/pixel_app_frame.bin`. The laptop's suite decodes a
  committed copy, `server/tests/fixtures/pixel_app_frame.bin`, with its real decoder.
- The laptop's suite writes `server/tests/fixtures/laptop_path.bin` from stated values, and the
  decoder test here reads that committed file and asserts the same values as literals. Run the
  laptop's tests first on a fresh clone, they write the file. Gradle hands the test the path, and
  `-PlaptopPathFixturePath=...` points it elsewhere.
- The timing log test writes `app/build/pixel_app_timing.jsonl`, and fails if the committed copy,
  `server/tests/fixtures/pixel_app_timing.jsonl`, no longer matches what the code writes. The
  laptop's timing report reads that committed copy. Regenerate it with
  `-PtimingFixturePath=<repo>/server/tests/fixtures/pixel_app_timing.jsonl`, an absolute path.

When any of these formats changes, regenerate its fixture on the writing side and commit it with
the change, and the reading side's test says whether the two still agree.

Nothing that needs an ARCore `Frame`, `Camera` or `Image`, the GL camera quad, or the composable
overlay has a JVM test. Those cannot be constructed off a device, so the phone is the test for
that layer, and the frame counts and the picture on screen are how it is read. The same goes for
the timing log's hooks in the renderer and the overlay. A walk's log shows they fire: a `frame`
line for every frame, and a `drawn` line for nearly every `received` one.
