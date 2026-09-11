package com.example.neuromorphicpaths.device

import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.core.session.DeviceSessionState
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed class ConnectionState {
    object Disconnected : ConnectionState()
    object Connecting : ConnectionState()
    object Connected : ConnectionState()
    data class Error(val message: String) : ConnectionState()
}

class WearableConnectionManager(private val scope: CoroutineScope) {
    private val _connectionState = MutableStateFlow<ConnectionState>(ConnectionState.Disconnected)
    val connectionState: StateFlow<ConnectionState> = _connectionState.asStateFlow()

    private var session: DeviceSession? = null
    private var connectionJob: Job? = null

    fun connect() {
        if (connectionJob?.isActive == true) return
        
        connectionJob = scope.launch {
            _connectionState.value = ConnectionState.Connecting
            
            withContext(Dispatchers.IO) {
                val result = Wearables.createSession(AutoDeviceSelector())
                val newSession = result.getOrNull()
                
                if (newSession == null) {
                    _connectionState.value = ConnectionState.Error("Failed to create wearable session")
                    return@withContext
                }
                
                session = newSession
                
                launch {
                    newSession.state.collect { state ->
                        when (state) {
                            DeviceSessionState.STARTED -> _connectionState.value = ConnectionState.Connected
                            DeviceSessionState.STOPPED -> _connectionState.value = ConnectionState.Disconnected
                            DeviceSessionState.PAUSED -> _connectionState.value = ConnectionState.Connecting
                            else -> Unit
                        }
                    }
                }
                
                newSession.start()
            }
        }
    }

    fun disconnect() {
        connectionJob?.cancel()
        session?.stop()
        session = null
        _connectionState.value = ConnectionState.Disconnected
    }
    
    internal fun getSession(): DeviceSession? = session
}
