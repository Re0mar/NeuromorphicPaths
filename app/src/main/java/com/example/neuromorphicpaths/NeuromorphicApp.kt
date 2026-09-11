package com.example.neuromorphicpaths

import android.app.Application
import com.example.neuromorphicpaths.device.CameraStreamManager
import com.example.neuromorphicpaths.device.WearableConnectionManager
import com.example.neuromorphicpaths.pipeline.FramePipeline
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

class NeuromorphicApp : Application() {
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    
    lateinit var connectionManager: WearableConnectionManager
    lateinit var cameraStreamManager: CameraStreamManager
    lateinit var framePipeline: FramePipeline

    override fun onCreate() {
        super.onCreate()
        instance = this
        
        connectionManager = WearableConnectionManager(applicationScope)
        cameraStreamManager = CameraStreamManager(applicationScope, connectionManager)
        framePipeline = FramePipeline(applicationScope, cameraStreamManager, this)
    }

    companion object {
        lateinit var instance: NeuromorphicApp
            private set
    }
}
