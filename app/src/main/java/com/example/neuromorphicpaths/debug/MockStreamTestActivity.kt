package com.example.neuromorphicpaths.debug

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.neuromorphicpaths.GlassesStreamService
import com.example.neuromorphicpaths.ui.theme.NeuromorphicPathsTheme
import com.meta.wearable.dat.mockdevice.MockDeviceKit
import com.meta.wearable.dat.mockdevice.api.GlassesModel
import com.meta.wearable.dat.mockdevice.api.MockDeviceKitInterface
import com.meta.wearable.dat.mockdevice.api.MockGlasses
import com.meta.wearable.dat.mockdevice.api.camera.CameraFacing
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class MockStreamTestActivity : ComponentActivity() {
    private val scope = CoroutineScope(Dispatchers.Main)
    private lateinit var mockDeviceKit: MockDeviceKitInterface
    private var mockGlasses: MockGlasses? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mockDeviceKit = MockDeviceKit.getInstance(this)

        setContent {
            NeuromorphicPathsTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MockStreamControls(
                        onStartMock = { startMockStream() },
                        onStopMock = { stopMockStream() },
                        onStartService = { startStreamingService() },
                        onStopService = { stopStreamingService() }
                    )
                }
            }
        }
    }

    private fun startMockStream() {
        scope.launch {
            mockDeviceKit.enable()
            mockDeviceKit.pairGlasses(GlassesModel.RAYBAN_META).fold(
                onSuccess = { glasses ->
                    mockGlasses = glasses
                    glasses.powerOn()
                    glasses.unfold()
                    glasses.don()
                    // Use phone camera as a mock feed for testing
                    glasses.services.camera.setCameraFeed(CameraFacing.FRONT)
                    Log.d("MockStreamTest", "Mock device ready and donning")
                },
                onFailure = { error, _ ->
                    Log.e("MockStreamTest", "Failed to pair mock glasses: $error")
                }
            )
        }
    }

    private fun stopMockStream() {
        scope.launch {
            mockGlasses?.powerOff()
            mockDeviceKit.disable()
            mockGlasses = null
        }
    }

    private fun startStreamingService() {
        val intent = Intent(this, GlassesStreamService::class.java)
        startForegroundService(intent)
    }

    private fun stopStreamingService() {
        val intent = Intent(this, GlassesStreamService::class.java)
        stopService(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopMockStream()
    }
}

@Composable
fun MockStreamControls(
    onStartMock: () -> Unit,
    onStopMock: () -> Unit,
    onStartService: () -> Unit,
    onStopService: () -> Unit
) {
    var isMockRunning by remember { mutableStateOf(false) }
    var isServiceRunning by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("Mock Stream Test", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(32.dp))

        Button(
            onClick = {
                if (isMockRunning) onStopMock() else onStartMock()
                isMockRunning = !isMockRunning
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (isMockRunning) "Stop Mock Device" else "Start Mock Device")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                if (isServiceRunning) onStopService() else onStartService()
                isServiceRunning = !isServiceRunning
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(
                containerColor = if (isServiceRunning) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.primary
            )
        ) {
            Text(if (isServiceRunning) "Stop Assistant Service" else "Start Assistant Service")
        }
        
        if (isServiceRunning) {
             Spacer(modifier = Modifier.height(32.dp))
             Text("Service is running. You can now open the Pipeline Overlay.", style = MaterialTheme.typography.bodySmall)
        }
    }
}
