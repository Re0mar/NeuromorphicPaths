package com.example.neuromorphicpaths.debug

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.neuromorphicpaths.NeuromorphicApp
import com.example.neuromorphicpaths.pipeline.PipelineFrame
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class PipelineOverlayViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application as NeuromorphicApp
    private val framePipeline = app.framePipeline
    private val logger = PipelineLogger(application)

    private val _latestFrame = MutableStateFlow<PipelineFrame?>(null)
    val latestFrame: StateFlow<PipelineFrame?> = _latestFrame.asStateFlow()

    private val _isLogging = MutableStateFlow(false)
    val isLogging: StateFlow<Boolean> = _isLogging.asStateFlow()

    init {
        viewModelScope.launch {
            framePipeline.processedFrames.collect { frame ->
                _latestFrame.value = frame
                if (_isLogging.value) {
                    logger.logFrame(frame)
                }
            }
        }
    }

    fun startPipeline() {
        app.connectionManager.connect()
        app.cameraStreamManager.startStreaming()
    }

    fun stopPipeline() {
        // We might not want to disconnect here if the service is using it,
        // but for the debug activity it's probably fine if it's the only one.
        // However, if the service is running, it should keep the connection.
    }

    fun toggleLogging() {
        if (_isLogging.value) {
            logger.stopLogging()
            _isLogging.value = false
        } else {
            logger.startLogging()
            _isLogging.value = true
        }
    }

    override fun onCleared() {
        super.onCleared()
        if (_isLogging.value) {
            logger.stopLogging()
        }
        stopPipeline()
    }
}
