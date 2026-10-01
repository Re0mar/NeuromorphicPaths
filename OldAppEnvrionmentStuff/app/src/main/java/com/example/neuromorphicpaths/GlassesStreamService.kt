package com.example.neuromorphicpaths

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.meta.wearable.dat.camera.addCamera
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.core.session.DeviceSessionState
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.StreamState
import com.meta.wearable.dat.camera.types.VideoQuality
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.time.Duration.Companion.milliseconds

class GlassesStreamService : Service() {
    private val serviceJob = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.Default + serviceJob)

    private var session: DeviceSession? = null
    private var sessionWatcherJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        startForeground(NOTIFICATION_ID, buildNotification())
        beginSessionLifecycle()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // START_STICKY: if the system kills the process for resources, try to
        // recreate the service (session state will be rebuilt from scratch).
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        sessionWatcherJob?.cancel()
        session?.stop()
        session = null
        scope.cancel()
        super.onDestroy()
    }

    /**
     * Owns the create -> start -> stream -> (stopped -> recreate) loop.
     * Runs for the lifetime of the service.
     */
    private fun beginSessionLifecycle() {
        sessionWatcherJob = scope.launch {
            sessionLoop@ while (true) {
                val result = Wearables.createSession(AutoDeviceSelector())
                val newSession = result.getOrNull()

                if (newSession == null) {
                    delay(RETRY_DELAY_MS.milliseconds)
                    continue@sessionLoop
                }

                session = newSession
                newSession.start()

                observeSessionAndStream(newSession)
            }
        }
    }

    private suspend fun observeSessionAndStream(session: DeviceSession) {
        var cameraStarted = false

        session.state.collect { state ->
            when (state) {
                DeviceSessionState.STARTED -> {
                    if (!cameraStarted) {
                        cameraStarted = true
                        startCameraStream(session)
                    }
                }
                DeviceSessionState.PAUSED -> {
                    // Glasses temporarily unavailable (hinges closed, gesture, etc)
                }
                DeviceSessionState.STOPPED-> {
                    // Session is dead; the outer loop will create a new one
                    return@collect
                }
                else -> Unit
            }
        }
    }

    private fun startCameraStream(session: DeviceSession) {
        val config = StreamConfiguration(videoQuality = VideoQuality.MEDIUM, frameRate = 24)

        session.addCamera(config).fold(
            onSuccess = { camera ->
                val stream = camera.stream

                scope.launch {
                    stream.videoStream.collect { frame ->
                        onFrame(frame)
                    }
                }

                scope.launch {
                    stream.state.collect { state ->
                        if (state == StreamState.STOPPED) {                        }
                    }
                }

                stream.start()
            },
            onFailure = { error, _ ->
            },
        )
    }

    private fun onFrame(frame: Any) {
        // TODO: Link the frames to our image recognizer
    }

    private fun buildNotification(): Notification {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Wearables camera stream",
            NotificationManager.IMPORTANCE_LOW,
        )
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Glasses camera active")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "wearables_stream"
        private const val RETRY_DELAY_MS = 3000L
    }
}