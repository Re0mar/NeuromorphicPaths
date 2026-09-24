# Implementation Plan - SidewalkVision Camera TFLite Inference & Overlay

This implementation plan outlines the steps required to update `MainActivity.kt` and add necessary dependencies, helper classes, and Compose UI components to stream live camera feed using CameraX, run the `best_int8.tflite` model (YOLO segmentation model with bounding box and mask outputs), and display the detections in real-time over the camera preview.

## User Review Required

> [!IMPORTANT]
> - **Model Inputs & Outputs**: The Python script shows an INT8 quantized YOLO model (`best_int8.tflite`) with letterbox preprocessing (114 padding, RGB float32/int8 quantization scaling and zero-point), and two outputs (detection tensor `[1, 116, 8400]` or similar, and prototype masks tensor `[1, 160, 160, 32]`). We will implement robust tensor extraction and dequantization supporting both float32 and quantized int8/uint8 tensors.
> - **Camera Permission**: Runtime camera permission request (`Manifest.permission.CAMERA`) will be integrated into `MainActivity`.
> - **Performance**: Running heavy TFLite inference (especially segmentation prototype mask multiplication + sigmoid) on every camera frame via `ImageAnalysis` requires running on a background executor (e.g. `Executors.newSingleThreadExecutor()`) and skipping frames if busy to avoid UI lag.

## Open Questions

- Should we structure the TFLite logic into a separate `PathDetector.kt` file for clean separation of concerns, keeping `MainActivity.kt` focused on camera lifecycle, permissions, and Compose UI? (Yes, this matches the Python comment `# mirror of the constant in PathDetector.kt` and provides clean architecture).

## Proposed Changes

### Build Configuration
#### [MODIFY] [build.gradle.kts](file:///C:/Users/danie/AndroidStudioProjects/SidewalkVision/app/build.gradle.kts)
- Add CameraX dependencies (`camera-camera2`, `camera-lifecycle`, `camera-view`).
- Add TensorFlow Lite / LiteRT dependencies (`org.tensorflow:tensorflow-lite` and optionally `tensorflow-lite-support`).
- Enable ViewBinding / data binding if needed, or stick to Jetpack Compose rendering.

---

### TFLite Inference Engine
#### [NEW] [PathDetector.kt](file:///C:/Users/danie/AndroidStudioProjects/SidewalkVision/app/src/main/java/com/example/sidewalkvision/PathDetector.kt)
- Load `best_int8.tflite` from assets using `MappedByteBuffer`.
- Implement letterbox scaling and padding matching the Python code (114, 114, 114 padding, aspect ratio preservation).
- Allocate tensors and prepare input tensor (handling quantization scale/zero-point if int8/uint8).
- Run `interpreter.invoke()`.
- Extract detection and prototype tensors, dequantize if necessary.
- Parse detection scores (`det[:, 4]`), find best anchor above confidence threshold (`CONF_THRESHOLD = 0.015`).
- Compute sigmoid mask multiplication (`proto @ coeffs`), apply threshold (`> 0.5`) and bounding box crop.
- Return detection results: bounding box (normalized coordinates) and segmentation bitmap/mask.

---

### UI & Camera Integration
#### [MODIFY] [MainActivity.kt](file:///C:/Users/danie/AndroidStudioProjects/SidewalkVision/app/src/main/java/com/example/sidewalkvision/MainActivity.kt)
- Request camera permission on startup.
- Set up CameraX `Preview` and `ImageAnalysis` use cases.
- Integrate `PathDetector` in the `ImageAnalysis` analyzer callback.
- Build a Jetpack Compose screen showing:
  1. `PreviewView` displaying the camera stream.
  2. A transparent overlay `Canvas` or custom composable drawing the green segmentation mask and red bounding box corresponding to detected walkways.
  3. Status overlay text ("Walkway detected: XX%" or "No walkway").

## Verification Plan

### Automated Tests
- Build verification via `./gradlew app:assembleDebug` to ensure all Kotlin files compile successfully and dependencies resolve correctly.

### Manual Verification
- Deploy the app to a connected Android device or emulator using `deploy()`.
- Grant camera permission when prompted.
- Verify that the camera feed is displayed, inference runs smoothly without crashing, and detection overlays appear correctly when pointing the camera at walkways/ground.
