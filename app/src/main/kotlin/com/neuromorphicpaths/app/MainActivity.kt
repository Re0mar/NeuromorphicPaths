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
import com.neuromorphicpaths.input.CameraXFrameSource
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
    private var pipelineJob: Job? = null
    private var activeSource: FrameSource? = null
    private var activeDetector: ObstacleDetector? = null

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startPipeline(cameraSource())
        }

    private val pickVideo =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) startPipeline(videoSource(uri))
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        poseProvider = SensorPoseProvider(this, heightMeters = AppDefaults.CAMERA_HEIGHT_METERS)
        poseProvider.start()
        val yoloWorldAvailable = OnnxYoloWorldDetector.isAvailable(this)
        setContent {
            val choice by detectorChoice.collectAsState()
            MaterialTheme {
                MainScreen(
                    display = screenDisplay,
                    detectorChoice = choice,
                    yoloWorldAvailable = yoloWorldAvailable,
                    onDetectorChoiceChange = { detectorChoice.value = it },
                    onStartCamera = { requestCameraPermission.launch(Manifest.permission.CAMERA) },
                    onOpenVideo = { pickVideo.launch(arrayOf("video/*")) },
                )
            }
        }
    }

    override fun onDestroy() {
        stopPipeline()
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

    private fun startPipeline(source: FrameSource) {
        stopPipeline()
        val detector = detector()
        activeSource = source
        activeDetector = detector
        val pipeline = GuidancePipeline(
            source = source,
            detector = detector,
            locator = GroundPlaneObstacleLocator(),
            field = PushFieldGuidance(),
            displays = listOf(screenDisplay, LogcatDisplay()),
        )
        pipelineJob = lifecycleScope.launch { pipeline.run() }
    }

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
