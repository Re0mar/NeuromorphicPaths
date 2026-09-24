# Fix Camera Preview Green Flickers and Optimize Path Vision

The goal is to fix the weird green flickers in the live preview and significantly improve performance by moving from a manual software-rendering pipeline to a hardware-accelerated direct rendering path.

## User Review Required

> [!IMPORTANT]
> The camera preview will now use hardware acceleration directly. This means the "weird green flickers" (caused by manual YUV conversion) will be completely resolved. The path detection overlay will be moved to a standard Compose layer, which is more performant and follows modern Android architecture.

## Proposed Changes

### Camera & Streaming

#### [MODIFY] [CameraViewModel.kt](file:///C:/Users/danie/AndroidStudioProjects/NeuromorphicPaths/app/src/main/java/com/example/neuromorphicpaths/camera/CameraViewModel.kt)
- Refactor `handleVideoFrame` to initialize the `HevcDecoder` using `targetSurface` directly for the preview.
- Remove the manual `drawProcessedFrame` logic that was converting YUV to Bitmaps and drawing to the canvas for the live display.
- Separate the path analysis logic so it runs on a dedicated background decoder/ImageReader only when `isVisionEnabled` is true.
- Optimize the YUV-to-Bitmap conversion used for TFLite analysis to avoid expensive JPEG compression.

#### [MODIFY] [CameraScreen.kt](file:///C:/Users/danie/AndroidStudioProjects/NeuromorphicPaths/app/src/main/java/com/example/neuromorphicpaths/ui/CameraScreen.kt)
- Add a Compose `Canvas` overlay to `PreviewBackground` to draw the detected path boundaries on top of the hardware-accelerated preview.
- Ensure the overlay correctly scales the normalized (0-1) points from the `PathDetector` to the screen dimensions.

## Verification Plan

### Manual Verification
- Deploy the app to the glasses and verify that the live preview is smooth and shows accurate colors (no green flickers) immediately upon starting the stream.
- Toggle "TensorFlow Path Vision" and verify that the path overlay appears correctly on top of the live video without causing lag or color changes in the background footage.
- Verify that photo capture and video recording still work as expected.
