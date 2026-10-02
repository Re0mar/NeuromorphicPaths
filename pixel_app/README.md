# Depth to laptop, the Pixel app

Written for whoever runs or extends the phone side. The app does one thing: it takes ARCore's
depth image, camera pose and floor plane off a Pixel and sends them to the laptop pipeline in
`server/` over TCP, in the format `server/docs/arcore_wire_format.md` describes. It draws
nothing. The walker reads the arrow from the laptop's web page in the phone's browser.

## Build

Android Studio's JDK and the Android SDK, both already on the build laptop. From this folder:

```
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ANDROID_HOME="$LOCALAPPDATA/Android/Sdk" ./gradlew.bat :app:assembleDebug
```

The debug APK lands in `app/build/outputs/apk/debug/`. The same toolchain versions as the group's
app on the `restart` branch: Gradle 9.6, AGP 9.4.1, Kotlin 2.2.10, minSdk 34.

## Run

1. Start the laptop listening: `python -m nav --source arcore_tcp --arcore-port 9000 --sink web`
   from `server/`, and open `http://<laptop>:8765` in the phone's browser for the arrow.
2. Install and open the app. It asks for the camera, then for ARCore if the phone lacks it.
3. Type the laptop's address, tap Connect. The two lines on screen say whether depth frames
   are flowing and whether the laptop is receiving them.

The phone and the laptop have to reach each other. On a phone hotspot the laptop is usually
`192.168.43.1`, which is the default on screen. On the emulator the laptop is `10.0.2.2`.

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
  z-forward ones, and `has_position` false whenever tracking is not `TRACKING`, because a stale
  position is worse than none.
- The lowest upward-facing plane as the floor, in the camera frame, or `null` when ARCore has
  none yet. The laptop fits its own floor in that case.
- Intrinsics scaled from the GPU camera texture to the depth image's size. The depth image
  covers the texture's 16:9 view, not the 4:3 CPU image, and scaling the wrong one made the
  horizontal focal length a third too long on the first run.
- One message per ARCore frame. The surface draws at the display rate, and a draw that gets the
  frame the previous draw already sent is skipped.

## Tests

`./gradlew.bat :app:testDebugUnitTest` runs the JVM tests: the message's own checks on its
shapes, the encoder, the connection against a loopback server, and the ARCore conversion's pure
functions (the quaternion, the floor plane, the DEPTH16 repacking). The encoder test also writes
`app/build/pixel_app_frame.bin`. The laptop's suite decodes a committed copy of that file,
`server/tests/fixtures/pixel_app_frame.bin`, with its real decoder. That cross-language test is
the contract check. When the format changes, regenerate the fixture and commit it with the change.

Nothing that needs an ARCore `Frame`, `Camera` or `Image` has a JVM test. Those classes cannot be
constructed off a device, so the phone is the test for that layer, and the frame counts on screen
are how it is read.
