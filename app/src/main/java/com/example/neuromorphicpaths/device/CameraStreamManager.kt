package com.example.neuromorphicpaths.device

import com.meta.wearable.dat.camera.addCamera
import com.meta.wearable.dat.camera.types.StreamConfiguration
import com.meta.wearable.dat.camera.types.VideoFrame
import com.meta.wearable.dat.camera.types.VideoQuality
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class CameraStreamManager(
    private val scope: CoroutineScope,
    private val connectionManager: WearableConnectionManager
) {
    private val _frames = MutableSharedFlow<VideoFrame>()
    val frames: Flow<VideoFrame> = _frames.asSharedFlow()

    private var streamingJob: Job? = null

    fun startStreaming() {
        if (streamingJob?.isActive == true) return
        
        val session = connectionManager.getSession() ?: return
        
        streamingJob = scope.launch {
            withContext(Dispatchers.IO) {
                val config = StreamConfiguration(videoQuality = VideoQuality.MEDIUM, frameRate = 24)
                session.addCamera(config).fold(
                    onSuccess = { camera ->
                        val stream = camera.stream
                        launch {
                            stream.videoStream.collect { frame ->
                                _frames.emit(frame)
                            }
                        }
                        stream.start()
                    },
                    onFailure = { _, _ ->
                        // Handle failure
                    }
                )
            }
        }
    }

    fun stopStreaming() {
        streamingJob?.cancel()
        streamingJob = null
    }
}
