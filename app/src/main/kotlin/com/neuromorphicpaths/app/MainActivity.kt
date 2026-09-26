package com.neuromorphicpaths.app

import android.Manifest
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.lifecycleScope
import com.neuromorphicpaths.app.ui.MainScreen
import com.neuromorphicpaths.core.Frame
import com.neuromorphicpaths.core.FrameSource
import com.neuromorphicpaths.core.GuidancePipeline
import com.neuromorphicpaths.core.ObstacleDetector
import com.neuromorphicpaths.core.SurfaceSegmenter
import com.neuromorphicpaths.core.WalkerState
import com.neuromorphicpaths.input.AccelerometerCadenceSpeed
import com.neuromorphicpaths.input.CameraXFrameSource
import com.neuromorphicpaths.input.GpsGroundSpeed
import com.neuromorphicpaths.input.Recording
import com.neuromorphicpaths.input.SensorPoseProvider
import com.neuromorphicpaths.input.VideoFileFrameSource
import com.neuromorphicpaths.math.DetectionTracker
import com.neuromorphicpaths.math.GroundPlaneObstacleLocator
import com.neuromorphicpaths.math.GroundPlaneSceneLocator
import com.neuromorphicpaths.math.PushFieldGuidance
import com.neuromorphicpaths.math.TrackedObstacleDetector
import com.neuromorphicpaths.math.TrackedObstacleLocator
import com.neuromorphicpaths.model.EmptyObstacleDetector
import com.neuromorphicpaths.model.OnnxSegFormerSegmenter
import com.neuromorphicpaths.model.OnnxYoloWorldDetector
import com.neuromorphicpaths.model.ScriptedObstacleDetector
import com.neuromorphicpaths.output.LogcatDisplay
import com.neuromorphicpaths.output.ScreenOverlayDisplay
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import java.io.File

/**
 * The composition root. Picks one implementation of each contract and runs the pipeline for
 * the lifetime of the screen. Nothing here knows how any stage works.
 */
class MainActivity : ComponentActivity() {

    private val screenDisplay = ScreenOverlayDisplay()
    private val detectorChoice = MutableStateFlow(DetectorChoice.NONE)
    private val segmenterOn = MutableStateFlow(false)
    private var segmenterAvailable = false
    // Set only from the adb intent, for timing comparisons between CPU and XNNPACK providers.
    private var segmenterUsesXnnpack = false
    private lateinit var poseProvider: SensorPoseProvider
    private lateinit var groundSpeed: GpsGroundSpeed
    private lateinit var cadenceSpeed: AccelerometerCadenceSpeed
    private var pipelineJob: Job? = null
    private var activeSource: FrameSource? = null
    private var activeDetector: ObstacleDetector? = null
    private var activeSegmenter: SurfaceSegmenter? = null

    // Remembered so a change of detector can restart the same kind of source with the same walker.
    private var activeSourceFactory: (() -> FrameSource)? = null
    private var activeWalkerState: (Frame) -> WalkerState = ::liveWalkerState

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            // Location is optional. Without it the walker's speed stays unknown and the field uses its default.
            if (granted[Manifest.permission.ACCESS_FINE_LOCATION] == true) groundSpeed.start()
            if (granted[Manifest.permission.CAMERA] == true) startPipeline(sourceFactory = { cameraSource() })
        }

    private val pickRecording =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
            val recording = uri?.let { Recording.fromTree(this, it) }
            if (recording != null) startRecording(recording) else Log.w(TAG, "No video in the picked folder")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        poseProvider = SensorPoseProvider(this, heightMeters = AppDefaults.CAMERA_HEIGHT_METERS)
        poseProvider.start()
        groundSpeed = GpsGroundSpeed(this)
        // Steps need no permission and work indoors, so they are the speed the field uses.
        cadenceSpeed = AccelerometerCadenceSpeed(this)
        cadenceSpeed.start()
        val yoloWorldAvailable = OnnxYoloWorldDetector.isAvailable(this)
        // The real detector is the point of the app, so it is the default whenever its model is bundled.
        if (yoloWorldAvailable) detectorChoice.value = DetectorChoice.YOLO_WORLD
        // The segmenter likewise runs whenever its model is bundled, and the adb intent can turn
        // it off or ask for the XNNPACK provider for a timing comparison:
        //   --ez segmenter false      --ez xnnpack true
        segmenterAvailable = OnnxSegFormerSegmenter.isAvailable(this)
        segmenterOn.value = segmenterAvailable && intent.getBooleanExtra(EXTRA_SEGMENTER, true)
        segmenterUsesXnnpack = intent.getBooleanExtra(EXTRA_XNNPACK, false)
        // adb hook for replaying a folder under files/recordings without touching the screen:
        //   adb shell am start -n com.neuromorphicpaths/.app.MainActivity --es recording outdoor1
        intent.getStringExtra(EXTRA_RECORDING)?.let { name ->
            val recording = Recording.fromDirectory(this, File(filesDir, "$RECORDINGS_DIR/$name"))
            if (recording != null) startRecording(recording) else Log.w(TAG, "No recording named $name under $RECORDINGS_DIR")
        }
        setContent {
            val choice by detectorChoice.collectAsState()
            val surfaces by segmenterOn.collectAsState()
            MaterialTheme {
                MainScreen(
                    display = screenDisplay,
                    detectorChoice = choice,
                    yoloWorldAvailable = yoloWorldAvailable,
                    segmenterOn = surfaces,
                    segmenterAvailable = segmenterAvailable,
                    onDetectorChoiceChange = { choice ->
                        detectorChoice.value = choice
                        // A running pipeline picks up the new detector at once rather than on the next start.
                        activeSourceFactory?.let { startPipeline(it, activeWalkerState) }
                    },
                    onSegmenterChange = { on ->
                        segmenterOn.value = on
                        activeSourceFactory?.let { startPipeline(it, activeWalkerState) }
                    },
                    onStartCamera = {
                        requestPermissions.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION))
                    },
                    onOpenRecording = { pickRecording.launch(null) },
                )
            }
        }
    }

    override fun onDestroy() {
        stopPipeline()
        cadenceSpeed.close()
        groundSpeed.close()
        poseProvider.close()
        screenDisplay.close()
        super.onDestroy()
    }

    private fun cameraSource(): FrameSource =
        CameraXFrameSource(this, this, poseProvider, AppDefaults.CAMERA_INTRINSICS)

    private fun startRecording(recording: Recording) {
        Log.i(TAG, "Replaying ${recording.videoUri.lastPathSegment}, logged pose: ${recording.hasLoggedPose}, logged speed: ${recording.hasLoggedSpeed}")
        val walker: (Frame) -> WalkerState = { frame ->
            val positionMillis = frame.timestampNanos / NANOS_PER_MILLI
            WalkerState(
                headingRadians = 0.0,
                // From the walker's steps in the acceleration log. Null without one, and the field keeps its default.
                speedMetersPerSecond = recording.speedForPosition(positionMillis),
                azimuthRadians = recording.azimuthForPosition(positionMillis),
            )
        }
        startPipeline(
            sourceFactory = {
                VideoFileFrameSource(
                    this,
                    recording.videoUri,
                    poseProvider,
                    AppDefaults.CAMERA_INTRINSICS,
                    poseForPosition = recording.poseForPosition(AppDefaults.CAMERA_HEIGHT_METERS),
                )
            },
            walkerState = walker,
        )
    }

    private fun detector(): ObstacleDetector = when (detectorChoice.value) {
        DetectorChoice.NONE -> EmptyObstacleDetector()
        DetectorChoice.SCRIPTED -> ScriptedObstacleDetector(AppDefaults.SCRIPTED_DETECTIONS)
        DetectorChoice.YOLO_WORLD -> if (OnnxYoloWorldDetector.isAvailable(this)) {
            OnnxYoloWorldDetector.load(this)
        } else {
            // The screen disables this choice when the model is missing, so this is a race, not a path.
            Log.w(TAG, "YOLO-World model asset missing, running without detections")
            EmptyObstacleDetector()
        }
    }

    private fun startPipeline(sourceFactory: () -> FrameSource, walkerState: (Frame) -> WalkerState = ::liveWalkerState) {
        stopPipeline()
        val source = sourceFactory()
        // One tracker per run, shared by the detector side that hands out ids and the locator
        // side that turns a track's range history into a closing speed.
        val tracker = DetectionTracker()
        val detector = TrackedObstacleDetector(detector(), tracker)
        val segmenter = segmenter()
        activeSourceFactory = sourceFactory
        activeWalkerState = walkerState
        activeSource = source
        activeDetector = detector
        activeSegmenter = segmenter
        val pipeline = GuidancePipeline(
            source = source,
            detector = detector,
            locator = TrackedObstacleLocator(GroundPlaneObstacleLocator(), tracker),
            field = PushFieldGuidance(),
            displays = listOf(screenDisplay, LogcatDisplay()),
            walkerState = walkerState,
            segmenter = segmenter,
            sceneLocator = segmenter?.let { GroundPlaneSceneLocator() },
        )
        pipelineJob = lifecycleScope.launch { pipeline.run() }
    }

    /** The surface segmenter when it is switched on and its model is bundled, otherwise none. */
    private fun segmenter(): SurfaceSegmenter? {
        if (!segmenterOn.value) return null
        if (!OnnxSegFormerSegmenter.isAvailable(this)) {
            // The switch is disabled when the model is missing, so this is a race, not a path.
            Log.w(TAG, "SegFormer model asset missing, running without surfaces")
            return null
        }
        Log.i(TAG, "Segmenter on, xnnpack: $segmenterUsesXnnpack")
        return OnnxSegFormerSegmenter.load(this, useXnnpack = segmenterUsesXnnpack)
    }

    /** Head forward, with whatever speed and azimuth the live sensors have measured so far. Steps first, GPS as the fallback. */
    @Suppress("UNUSED_PARAMETER")
    private fun liveWalkerState(frame: Frame): WalkerState = WalkerState(
        headingRadians = 0.0,
        speedMetersPerSecond = cadenceSpeed.metersPerSecond.value ?: groundSpeed.metersPerSecond.value,
        azimuthRadians = poseProvider.currentAzimuthRadians(),
    )

    private fun stopPipeline() {
        pipelineJob?.cancel()
        pipelineJob = null
        activeSource?.close()
        activeSource = null
        activeDetector?.close()
        activeDetector = null
        activeSegmenter?.close()
        activeSegmenter = null
    }

    private companion object {
        const val TAG = "MainActivity"
        const val EXTRA_RECORDING = "recording"
        const val EXTRA_SEGMENTER = "segmenter"
        const val EXTRA_XNNPACK = "xnnpack"
        const val RECORDINGS_DIR = "recordings"
        const val NANOS_PER_MILLI = 1_000_000L
    }
}
