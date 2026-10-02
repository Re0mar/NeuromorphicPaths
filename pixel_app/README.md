# Depth to laptop, the Pixel app

Written for whoever runs or extends the phone side. The app takes ARCore's depth image, camera
pose and floor plane off a Pixel and sends them to the laptop pipeline in `server/` over TCP, in
the format `server/docs/arcore_wire_format.md` describes. The laptop plans a path and sends it
back over a second connection, and the app draws the camera picture with an arrow over it
pointing where the path says. The walker looks at the phone and nothing else.

## Build

Android Studio's JDK and the Android SDK, both already on the build laptop. From this folder:

```
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ANDROID_HOME="$LOCALAPPDATA/Android/Sdk" ./gradlew.bat :app:assembleDebug
```

The debug APK lands in `app/build/outputs/apk/debug/`. The same toolchain versions as the group's
app on the `restart` branch: Gradle 9.6, AGP 9.4.1, Kotlin 2.2.10, minSdk 34.

## Run

1. Start the laptop listening, from `server/`:
   `python -m nav --source arcore_tcp --arcore-accept-timeout 600 --reconnect --sink phone_app --sink web --floor-max-tilt 50`.
   That serves the arrow to this app and the depth view to a browser at `http://<laptop>:8765` in
   the same run. It listens on three ports, 9000 for depth, 9100 for paths and 8765 for the page.
   Each needs an inbound firewall rule on the laptop, and the rule has to name the Python that
   owns the socket, which with a venv is the base interpreter and not the venv's launcher. See
   `server/README.md`.
2. Install and open the app. It asks for the camera, then for ARCore if the phone lacks it.
3. Type the laptop's address, leave the two ports unless the laptop was started with others,
   and tap Connect. The app opens both connections to that one address.

What the screen shows, top to bottom: the three status lines, then the camera picture with the
arrow over it.

- Line one: the depth connection, frames sent and frames dropped as stale.
- Line two: whether ARCore is tracking, how many frames had depth, and whether a floor is found.
- Line three: the path connection, paths received and refused, and the last refusal's reason.
- The arrow points where the newest path says, positive to the right, with the heading in
  degrees under it. Green normally, red with `ALARM` when something is under a second from
  contact, grey with `No path yet` before the first path. The age under the heading counts up
  from when the path arrived and says `stale` after a second without a new one, so a frozen
  arrow is visibly frozen.

The phone and the laptop have to reach each other. On a phone hotspot the laptop is usually
`192.168.43.1`, which is the default on screen. On the emulator the laptop is `10.0.2.2`. Over
USB, `adb reverse tcp:9000 tcp:9000` and `adb reverse tcp:9100 tcp:9100` make `127.0.0.1` on the
phone reach the laptop. Started from adb, the app connects by itself:

```
adb shell am start -n com.neuromorphicpaths.pixel/.MainActivity --es host 127.0.0.1 --ei port 9000 --ei path_port 9100
```

Installing on the emulator from this machine: use `adb push` and `pm install`, not
`adb install`. The streamed install wedges the emulator's package manager.

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

## Tests

`./gradlew.bat :app:testDebugUnitTest` runs the JVM tests: the two messages' own checks on their
shapes, the encoder, the path decoder one rule per test, both connections against loopback
servers, the floor choice, the arrow's arithmetic, and the ARCore conversion's pure functions
(the quaternion, the floor plane, the DEPTH16 repacking). Two of them are the contract checks
across the language boundary:

- The encoder test writes `app/build/pixel_app_frame.bin`. The laptop's suite decodes a
  committed copy, `server/tests/fixtures/pixel_app_frame.bin`, with its real decoder.
- The laptop's suite writes `server/tests/fixtures/laptop_path.bin` from stated values, and the
  decoder test here reads that committed file and asserts the same values as literals. Run the
  laptop's tests first on a fresh clone, they write the file. Gradle hands the test the path, and
  `-PlaptopPathFixturePath=...` points it elsewhere.

When either format changes, regenerate the fixture on the writing side and commit it with the
change, and the reading side's test says whether the two still agree.

Nothing that needs an ARCore `Frame`, `Camera` or `Image`, the GL camera quad, or the composable
overlay has a JVM test. Those cannot be constructed off a device, so the phone is the test for
that layer, and the frame counts and the picture on screen are how it is read.
