package com.neuromorphicpaths.app

import android.Manifest
import android.net.Uri
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
import com.neuromorphicpaths.core.FrameSource
import com.neuromorphicpaths.core.GuidancePipeline
import com.neuromorphicpaths.core.ObstacleDetector
import com.neuromorphicpaths.core.WalkerState
import com.neuromorphicpaths.input.CameraXFrameSource
import com.neuromorphicpaths.input.GpsGroundSpeed
import com.neuromorphicpaths.input.SensorPoseProvider
import com.neuromorphicpaths.input.VideoFileFrameSource
import com.neuromorphicpaths.math.GroundPlaneObstacleLocator
import com.neuromorphicpaths.math.PushFieldGuidance
import com.neuromorphicpaths.model.EmptyObstacleDetector
import com.neuromorphicpaths.model.OnnxYoloWorldDetector
import com.neuromorphicpaths.model.ScriptedObstacleDetector
import com.neuromorphicpaths.output.LogcatDisplay
import com.neuromorphicpaths.output.ScreenOverlayDisplay
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

/**
 * The composition root. Picks one implementation of each contract and runs the pipeline for
 * the lifetime of the screen. Nothing here knows how any stage works.
 */
class MainActivity : ComponentActivity() {

    private val screenDisplay = ScreenOverlayDisplay()
    private val detectorChoice = MutableStateFlow(DetectorChoice.NONE)
    private lateinit var poseProvider: SensorPoseProvider
    private lateinit var groundSpeed: GpsGroundSpeed
    private var pipelineJob: Job? = null
    private var activeSource: FrameSource? = null
    private var activeDetector: ObstacleDetector? = null

    // Remembered so a change of detector can restart the same kind of source.
    private var activeSourceFactory: (() -> FrameSource)? = null

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            // Location is optional. Without it the walker's speed stays unknown and the field uses its default.
            if (granted[Manifest.permission.ACCESS_FINE_LOCATION] == true) groundSpeed.start()
            if (granted[Manifest.permission.CAMERA] == true) startPipeline { cameraSource() }
        }

    private val pickVideo =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) startPipeline { videoSource(uri) }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        poseProvider = SensorPoseProvider(this, heightMeters = AppDefaults.CAMERA_HEIGHT_METERS)
        poseProvider.start()
        groundSpeed = GpsGroundSpeed(this)
        val yoloWorldAvailable = OnnxYoloWorldDetector.isAvailable(this)
        // The real detector is the point of the app, so it is the default whenever its model is bundled.
        if (yoloWorldAvailable) detectorChoice.value = DetectorChoice.YOLO_WORLD
        setContent {
            val choice by detectorChoice.collectAsState()
            MaterialTheme {
                MainScreen(
                    display = screenDisplay,
                    detectorChoice = choice,
                    yoloWorldAvailable = yoloWorldAvailable,
                    onDetectorChoiceChange = { choice ->
                        detectorChoice.value = choice
                        // A running pipeline picks up the new detector at once rather than on the next start.
                        activeSourceFactory?.let { startPipeline(it) }
                    },
                    onStartCamera = {
                        requestPermissions.launch(arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION))
                    },
                    onOpenVideo = { pickVideo.launch(arrayOf("video/*")) },
                )
            }
        }
    }

    override fun onDestroy() {
        stopPipeline()
        groundSpeed.close()
        poseProvider.close()
        screenDisplay.close()
        super.onDestroy()
    }

    private fun cameraSource(): FrameSource =
        CameraXFrameSource(this, this, poseProvider, AppDefaults.CAMERA_INTRINSICS)

    private fun videoSource(uri: Uri): FrameSource =
        VideoFileFrameSource(this, uri, poseProvider, AppDefaults.CAMERA_INTRINSICS)

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

    private fun startPipeline(sourceFactory: () -> FrameSource) {
        stopPipeline()
        val source = sourceFactory()
        val detector = detector()
        activeSourceFactory = sourceFactory
        activeSource = source
        activeDetector = detector
        val pipeline = GuidancePipeline(
            source = source,
            detector = detector,
            locator = GroundPlaneObstacleLocator(),
            field = PushFieldGuidance(),
            displays = listOf(screenDisplay, LogcatDisplay()),
            walkerState = ::currentWalkerState,
        )
        pipelineJob = lifecycleScope.launch { pipeline.run() }
    }

    /** Head forward, with whatever speed and azimuth the sensors have measured so far. */
    private fun currentWalkerState(): WalkerState = WalkerState(
        headingRadians = 0.0,
        speedMetersPerSecond = groundSpeed.metersPerSecond.value,
        azimuthRadians = poseProvider.currentAzimuthRadians(),
    )

    private fun stopPipeline() {
        pipelineJob?.cancel()
        pipelineJob = null
        activeSource?.close()
        activeSource = null
        activeDetector?.close()
        activeDetector = null
    }

    private companion object {
        const val TAG = "MainActivity"
    }
}
