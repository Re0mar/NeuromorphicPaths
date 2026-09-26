package com.neuromorphicpaths.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import androidx.lifecycle.lifecycleScope
import androidx.savedstate.SavedStateRegistry
import androidx.savedstate.SavedStateRegistryController
import androidx.savedstate.SavedStateRegistryOwner
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
import com.neuromorphicpaths.output.FloatingGuidanceDisplay
import com.neuromorphicpaths.output.LogcatDisplay
import com.neuromorphicpaths.output.ScreenOverlayDisplay
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/** What the pipeline is reading from, if anything. */
enum class GuidanceSource {
    CAMERA,
    RECORDING,
}

/**
 * The composition root. Picks one implementation of each contract and runs the pipeline in a
 * foreground service, so it keeps running while another app is in front and the floating
 * overlay draws over that app. Nothing here knows how any stage works.
 *
 * The screen binds to it for the controls and for the display it draws. Android only lets a
 * camera service start while the app has a visible screen, so the pipeline is always started
 * from the screen, and after that the person can switch away.
 */
class GuidanceService : LifecycleService(), SavedStateRegistryOwner {

    inner class LocalBinder : Binder() {
        val service: GuidanceService get() = this@GuidanceService
    }

    private val binder = LocalBinder()
    private val savedStateController = SavedStateRegistryController.create(this)
    override val savedStateRegistry: SavedStateRegistry get() = savedStateController.savedStateRegistry

    val screenDisplay = ScreenOverlayDisplay()
    lateinit var floatingDisplay: FloatingGuidanceDisplay
        private set

    private val detectorChoiceState = MutableStateFlow(DetectorChoice.NONE)
    val detectorChoice: StateFlow<DetectorChoice> = detectorChoiceState.asStateFlow()
    private val segmenterOnState = MutableStateFlow(false)
    val segmenterOn: StateFlow<Boolean> = segmenterOnState.asStateFlow()
    private val runningSourceState = MutableStateFlow<GuidanceSource?>(null)
    val runningSource: StateFlow<GuidanceSource?> = runningSourceState.asStateFlow()

    var yoloWorldAvailable = false
        private set
    var segmenterAvailable = false
        private set

    // Set only from the adb intent, for timing comparisons between CPU and XNNPACK providers.
    var segmenterUsesXnnpack = false

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

    override fun onCreate() {
        // The saved-state registry has to be attached before the lifecycle leaves INITIALIZED,
        // which happens inside super.onCreate.
        savedStateController.performAttach()
        savedStateController.performRestore(null)
        super.onCreate()
        poseProvider = SensorPoseProvider(this, heightMeters = AppDefaults.CAMERA_HEIGHT_METERS)
        poseProvider.start()
        groundSpeed = GpsGroundSpeed(this)
        // Steps need no permission and work indoors, so they are the speed the field uses.
        cadenceSpeed = AccelerometerCadenceSpeed(this)
        cadenceSpeed.start()
        yoloWorldAvailable = OnnxYoloWorldDetector.isAvailable(this)
        // The real detector is the point of the app, so it is the default whenever its model is bundled.
        if (yoloWorldAvailable) detectorChoiceState.value = DetectorChoice.YOLO_WORLD
        // The segmenter likewise runs whenever its model is bundled. The screen can turn it off.
        segmenterAvailable = OnnxSegFormerSegmenter.isAvailable(this)
        segmenterOnState.value = segmenterAvailable
        floatingDisplay = FloatingGuidanceDisplay(this, this, this, lifecycleScope)
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Guidance running", NotificationManager.IMPORTANCE_LOW),
        )
    }

    override fun onBind(intent: Intent): IBinder {
        super.onBind(intent)
        return binder
    }

    override fun onDestroy() {
        stopPipeline()
        cadenceSpeed.close()
        groundSpeed.close()
        poseProvider.close()
        screenDisplay.close()
        floatingDisplay.close()
        super.onDestroy()
    }

    /** Starts the ground speed from GPS. Only after the screen has the location permission. */
    fun startGroundSpeed() = groundSpeed.start()

    /** Starts the live camera. Only after the screen has the camera permission, and only while it is visible. */
    fun startCamera() {
        goForeground(ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA)
        runningSourceState.value = GuidanceSource.CAMERA
        startPipeline(sourceFactory = { CameraXFrameSource(this, this, poseProvider, AppDefaults.CAMERA_INTRINSICS) })
    }

    fun startRecording(recording: Recording) {
        Log.i(TAG, "Replaying ${recording.videoUri.lastPathSegment}, logged pose: ${recording.hasLoggedPose}, logged speed: ${recording.hasLoggedSpeed}")
        // Media processing is the type meant for work on a video file. It only exists from
        // Android 15, and on 14 data sync is the nearest one allowed without a camera.
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.VANILLA_ICE_CREAM) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        } else {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        }
        goForeground(type)
        runningSourceState.value = GuidanceSource.RECORDING
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

    fun setDetectorChoice(choice: DetectorChoice) {
        detectorChoiceState.value = choice
        // A running pipeline picks up the new detector at once rather than on the next start.
        activeSourceFactory?.let { startPipeline(it, activeWalkerState) }
    }

    fun setSegmenterOn(on: Boolean) {
        segmenterOnState.value = on && segmenterAvailable
        activeSourceFactory?.let { startPipeline(it, activeWalkerState) }
    }

    /** Stops the pipeline, takes the notification down, and lets the service end once the screen unbinds. */
    fun stopGuidance() {
        stopPipeline()
        activeSourceFactory = null
        runningSourceState.value = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    // A bound service only survives its screen going away once it is also started, so it starts
    // itself before promoting. Calling this again with another type switches the type in place.
    // The platform call, not ServiceCompat's: on the Pixel with Android 17 that one sent type 0
    // for media processing, a type newer than Android 14, and the system refused it.
    private fun goForeground(type: Int) {
        ContextCompat.startForegroundService(this, Intent(this, GuidanceService::class.java))
        startForeground(NOTIFICATION_ID, notification(), type)
    }

    private fun notification(): Notification {
        val openScreen = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_directions)
            .setContentTitle("NeuromorphicPaths is guiding")
            .setContentText("Tap to open the controls")
            .setContentIntent(openScreen)
            .setOngoing(true)
            .build()
    }

    private fun detector(): ObstacleDetector = when (detectorChoiceState.value) {
        DetectorChoice.NONE -> EmptyObstacleDetector()
        DetectorChoice.SCRIPTED -> ScriptedObstacleDetector(AppDefaults.SCRIPTED_DETECTIONS)
        DetectorChoice.YOLO_WORLD -> if (yoloWorldAvailable) {
            OnnxYoloWorldDetector.load(this)
        } else {
            // The screen disables this choice when the model is missing, so this is a race, not a path.
            Log.w(TAG, "YOLO-World model asset missing, running without detections")
            EmptyObstacleDetector()
        }
    }

    /** The surface segmenter when it is switched on and its model is bundled, otherwise none. */
    private fun segmenter(): SurfaceSegmenter? {
        if (!segmenterOnState.value) return null
        if (!segmenterAvailable) {
            // The switch is disabled when the model is missing, so this is a race, not a path.
            Log.w(TAG, "SegFormer model asset missing, running without surfaces")
            return null
        }
        Log.i(TAG, "Segmenter on, xnnpack: $segmenterUsesXnnpack")
        return OnnxSegFormerSegmenter.load(this, useXnnpack = segmenterUsesXnnpack)
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
            displays = listOf(screenDisplay, floatingDisplay, LogcatDisplay()),
            walkerState = walkerState,
            segmenter = segmenter,
            sceneLocator = segmenter?.let { GroundPlaneSceneLocator() },
        )
        pipelineJob = lifecycleScope.launch { pipeline.run() }
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
        const val TAG = "GuidanceService"
        const val CHANNEL_ID = "guidance"
        const val NOTIFICATION_ID = 1
        const val NANOS_PER_MILLI = 1_000_000L
    }
}
