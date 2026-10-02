package com.example.sidewalkvision

import android.content.Context
import android.os.Handler
import android.os.Looper
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Enumeration of available vision hardware sources.
 */
enum class VisionSource {
    ATTACHED_CAMERA,
    PUPIL_NEON_GLASSES
}

/**
 * Real-time gaze point container representing normalized eye position in scene camera space.
 * x: [0.0..1.0] from left to right
 * y: [0.0..1.0] from top to bottom
 */
data class GazePoint(
    val x: Float,
    val y: Float,
    val confidence: Float = 1.0f,
    val timestampMs: Long = System.currentTimeMillis(),
    val isSimulated: Boolean = false
)

/**
 * Manager handling connection, real-time streaming, filtering, and fallback simulation
 * for Pupil Labs Neon Smartglasses over local network / Companion API.
 */
class PupilNeonManager(private val context: Context) {

    companion object {
        private const val TAG = "PupilNeonManager"
        private const val SIM_INTERVAL_MS = 33L // ~30 fps
    }

    var host: String = AppConfig.PupilNeon.DEFAULT_HOST
    var port: Int = AppConfig.PupilNeon.DEFAULT_PORT

    @Volatile private var isRunning = false
    @Volatile private var isConnected = false
    @Volatile private var useSimulation = true
    @Volatile private var currentGazePoint: GazePoint? = null

    private var smoothedX = 0.5f
    private var smoothedY = 0.5f

    private val executor: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val gazeListeners = mutableListOf<(GazePoint) -> Unit>()

    // Simulation state for natural gaze trajectories
    private var simTargetX = 0.5f
    private var simTargetY = 0.4f
    private var simAngle = 0f
    private var lastFixationChangeMs = System.currentTimeMillis()

    fun start() {
        if (isRunning) return
        isRunning = true
        executor.scheduleWithFixedDelay({
            pollGazeOrSimulate()
        }, 0, SIM_INTERVAL_MS, TimeUnit.MILLISECONDS)
    }

    fun stop() {
        isRunning = false
        isConnected = false
    }

    fun addGazeListener(listener: (GazePoint) -> Unit) {
        gazeListeners.add(listener)
    }

    fun removeGazeListener(listener: (GazePoint) -> Unit) {
        gazeListeners.remove(listener)
    }

    fun getCurrentGaze(): GazePoint? = currentGazePoint

    fun isConnected(): Boolean = isConnected

    fun isSimulated(): Boolean = useSimulation

    fun setSimulationMode(enabled: Boolean) {
        useSimulation = enabled
        if (enabled) {
            isConnected = false
        }
    }

    fun getStatusText(): String = when {
        !isRunning -> "OFFLINE"
        isConnected -> "CONNECTED ($host)"
        useSimulation -> "SIMULATING GAZE"
        else -> "CONNECTING TO $host..."
    }

    private fun pollGazeOrSimulate() {
        if (!isRunning) return

        if (!useSimulation) {
            val liveGaze = tryFetchLiveGaze()
            if (liveGaze != null) {
                isConnected = true
                processGaze(liveGaze.x, liveGaze.y, liveGaze.confidence, isSim = false)
                return
            } else {
                isConnected = false
            }
        }

        // Fallback or explicit simulation
        updateSimulatedGaze()
    }

    private fun tryFetchLiveGaze(): GazePoint? {
        return try {
            // Attempt socket/HTTP check against Pupil Labs Companion API endpoint
            val url = URL("http://$host:$port/api/gaze")
            val conn = url.openConnection() as HttpURLConnection
            conn.connectTimeout = 500
            conn.readTimeout = 500
            conn.requestMethod = "GET"

            if (conn.responseCode == 200) {
                val response = conn.inputStream.bufferedReader().use { it.readText() }
                parseGazeJson(response)
            } else {
                conn.disconnect()
                null
            }
        } catch (e: Exception) {
            null
        }
    }

    private fun parseGazeJson(jsonStr: String): GazePoint? {
        // Simple regex-based JSON parser to extract x, y, confidence without external dependencies
        return try {
            val xMatch = Regex(""""x"\s*:\s*([0-9.]+)""").find(jsonStr)
            val yMatch = Regex(""""y"\s*:\s*([0-9.]+)""").find(jsonStr)
            val confMatch = Regex(""""confidence"\s*:\s*([0-9.]+)""").find(jsonStr)

            if (xMatch != null && yMatch != null) {
                val rawX = xMatch.groupValues[1].toFloat()
                val rawY = yMatch.groupValues[1].toFloat()
                val conf = confMatch?.groupValues?.get(1)?.toFloat() ?: 1.0f
                GazePoint(x = rawX.coerceIn(0f, 1f), y = rawY.coerceIn(0f, 1f), confidence = conf)
            } else null
        } catch (e: Exception) {
            null
        }
    }

    private fun updateSimulatedGaze() {
        val now = System.currentTimeMillis()

        // Switch fixation targets periodically to simulate human visual exploration
        if (now - lastFixationChangeMs > 1800) {
            lastFixationChangeMs = now
            simTargetX = 0.25f + Random.nextFloat() * 0.50f
            simTargetY = 0.30f + Random.nextFloat() * 0.40f
        }

        simAngle += 0.08f
        val orbitalOffsetX = 0.04f * sin(simAngle)
        val orbitalOffsetY = 0.03f * cos(simAngle * 0.7f)

        val targetX = (simTargetX + orbitalOffsetX).coerceIn(0.1f, 0.9f)
        val targetY = (simTargetY + orbitalOffsetY).coerceIn(0.1f, 0.9f)

        processGaze(targetX, targetY, confidence = 0.95f, isSim = true)
    }

    private fun processGaze(rawX: Float, rawY: Float, confidence: Float, isSim: Boolean) {
        val alpha = AppConfig.PupilNeon.GAZE_ALPHA
        smoothedX += alpha * (rawX - smoothedX)
        smoothedY += alpha * (rawY - smoothedY)

        val gaze = GazePoint(
            x = smoothedX,
            y = smoothedY,
            confidence = confidence,
            isSimulated = isSim
        )
        currentGazePoint = gaze

        mainHandler.post {
            for (listener in gazeListeners) {
                listener(gaze)
            }
        }
    }
}
