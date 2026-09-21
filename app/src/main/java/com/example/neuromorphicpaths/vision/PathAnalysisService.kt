package com.example.neuromorphicpaths.vision

import android.app.Service
import android.content.Intent
import android.graphics.Bitmap
import android.os.Binder
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * PathAnalysisService receives camera frames and runs image recognition
 * to detect and track walkways.
 */
class PathAnalysisService : Service() {

    companion object {
        private const val TAG = "PathAnalysisService"
    }

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.Default + serviceJob)

    private lateinit var pathDetector: PathDetector

    private val _detectionResult = MutableStateFlow<PathResult?>(null)
    val detectionResult: StateFlow<PathResult?> = _detectionResult.asStateFlow()

    inner class LocalBinder : Binder() {
        fun getService(): PathAnalysisService = this@PathAnalysisService
    }

    private val binder = LocalBinder()

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        pathDetector = PathDetector(this)
    }

    private var isProcessing = false

    /**
     * Submits a frame for processing. This is non-blocking and skips frames if busy.
     */
    fun processFrame(bitmap: Bitmap) {
        if (isProcessing) return
        isProcessing = true
        
        serviceScope.launch {
            try {
                val result = pathDetector.detectPath(bitmap)
                _detectionResult.value = result
            } catch (e: Exception) {
                Log.e(TAG, "Error processing frame", e)
            } finally {
                isProcessing = false
            }
        }
    }

    override fun onDestroy() {
        Log.d(TAG, "Service destroyed")
        pathDetector.close()
        serviceJob.cancel()
        super.onDestroy()
    }
}
