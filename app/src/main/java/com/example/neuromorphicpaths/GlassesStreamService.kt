package com.example.neuromorphicpaths

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.example.neuromorphicpaths.device.ConnectionState
import com.example.neuromorphicpaths.display.AudioFeedbackManager
import com.example.neuromorphicpaths.display.FeedbackRouter
import com.example.neuromorphicpaths.display.GlassesDisplayRenderer
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class GlassesStreamService : Service() {
    private val serviceJob = SupervisorJob()
    private val scope = CoroutineScope(Dispatchers.Default + serviceJob)

    private var audioFeedbackManager: AudioFeedbackManager? = null
    private var displayRenderer: GlassesDisplayRenderer? = null
    private var feedbackRouter: FeedbackRouter? = null

    override fun onCreate() {
        super.onCreate()
        audioFeedbackManager = AudioFeedbackManager(this)
        
        startForeground(NOTIFICATION_ID, buildNotification())
        
        val app = NeuromorphicApp.instance
        
        // Start connection and streaming
        app.connectionManager.connect()
        app.cameraStreamManager.startStreaming()
        
        // Observe connection to setup display renderer
        scope.launch {
            app.connectionManager.connectionState.collect { state ->
                if (state is ConnectionState.Connected) {
                    val session = app.connectionManager.getSession()
                    if (session != null) {
                        displayRenderer = GlassesDisplayRenderer(scope, session)
                        feedbackRouter = FeedbackRouter(displayRenderer!!, audioFeedbackManager!!)
                    }
                } else if (state is ConnectionState.Disconnected) {
                    displayRenderer?.stop()
                    displayRenderer = null
                    feedbackRouter = null
                }
            }
        }

        // Observe pipeline for results
        scope.launch {
            app.framePipeline.processedFrames.collect { pipelineFrame ->
                pipelineFrame.stabilizedCommand?.let { command ->
                    feedbackRouter?.routeFeedback(command, pipelineFrame.timestampMs)
                }
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        displayRenderer?.stop()
        displayRenderer = null
        feedbackRouter = null
        audioFeedbackManager?.shutdown()
        audioFeedbackManager = null
        
        val app = NeuromorphicApp.instance
        app.cameraStreamManager.stopStreaming()
        app.connectionManager.disconnect()
        
        scope.cancel()
        super.onDestroy()
    }

    private fun buildNotification(): Notification {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Sidewalk Navigation Assistant",
            NotificationManager.IMPORTANCE_LOW,
        )
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Sidewalk Assistant Active")
            .setContentText("Monitoring sidewalk and providing navigation cues.")
            .setSmallIcon(R.drawable.camera_access_icon)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val NOTIFICATION_ID = 1002
        private const val CHANNEL_ID = "sidewalk_assistant_stream"
    }
}
