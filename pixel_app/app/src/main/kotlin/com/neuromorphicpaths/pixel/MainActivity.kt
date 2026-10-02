package com.neuromorphicpaths.pixel

import android.Manifest
import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.FatalException
import com.google.ar.core.exceptions.UnavailableException
import com.neuromorphicpaths.pixel.ar.CaptureState
import com.neuromorphicpaths.pixel.ar.DepthCaptureRenderer
import com.neuromorphicpaths.pixel.wire.ConnectionStatus
import com.neuromorphicpaths.pixel.wire.LaptopConnection
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Sends the Pixel's ARCore depth to the laptop pipeline. One screen, two numbers that matter:
 * whether depth frames are flowing and whether the laptop is receiving them.
 */
class MainActivity : ComponentActivity() {
    private var session: Session? = null
    private var connection: LaptopConnection? = null
    private var surface: GLSurfaceView? = null
    private val captureState = MutableStateFlow<CaptureState>(CaptureState.CameraUnavailable)
    private val connectionStatus = MutableStateFlow<ConnectionStatus>(ConnectionStatus.Disconnected("not started"))
    private var installRequested = false
    private var startupHost = DEFAULT_HOST
    private var startupPort = DEFAULT_PORT

    private val cameraPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startSession() else connectionStatus.value = ConnectionStatus.Disconnected("camera permission refused")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Started from adb with --es host and --ei port, the app connects by itself. That is how
        // the emulator run drives it with nobody typing into a headless screen.
        val hostExtra = intent.getStringExtra(EXTRA_HOST)
        if (hostExtra != null) {
            startupHost = hostExtra
            startupPort = intent.getIntExtra(EXTRA_PORT, DEFAULT_PORT)
            connect(startupHost, startupPort)
        }
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Screen(
                        captureState = captureState,
                        connectionStatus = connectionStatus,
                        onConnect = { host, port -> connect(host, port) },
                        glSurface = { view -> surface = view },
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startSession()
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA)
        }
        surface?.onResume()
    }

    override fun onPause() {
        super.onPause()
        surface?.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        connection?.stop()
        session?.close()
        session = null
        super.onDestroy()
    }

    private fun startSession() {
        if (session != null) {
            session?.resume()
            return
        }
        try {
            // ARCore may need installing or updating. requestInstall returns INSTALL_REQUESTED the
            // first time and the activity resumes again once that is done, so ask at most once.
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    installRequested = true
                    return
                }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }
            val created = Session(this)
            val config = created.config
            // Depth is the whole point. AUTOMATIC is the smoothed depth ARCore meant for apps,
            // RAW_DEPTH_ONLY is sparser. A device without depth support is refused rather than
            // sent along without it.
            if (!created.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) {
                connectionStatus.value = ConnectionStatus.Disconnected("this device has no ARCore Depth API")
                created.close()
                return
            }
            config.depthMode = Config.DepthMode.AUTOMATIC
            config.planeFindingMode = Config.PlaneFindingMode.HORIZONTAL
            config.updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            config.focusMode = Config.FocusMode.AUTO
            created.configure(config)
            created.resume()
            session = created
        } catch (unavailable: UnavailableException) {
            // ARCore not installed, too old, or this device is not supported. Nothing to retry.
            Log.e(TAG, "ARCore unavailable", unavailable)
            connectionStatus.value = ConnectionStatus.Disconnected("ARCore unavailable: ${unavailable.javaClass.simpleName}")
        } catch (fatal: FatalException) {
            // ARCore itself failed to start. Seen on the emulator under a software GPU. Not an
            // UnavailableException, so without this the app died on resume with no message.
            Log.e(TAG, "ARCore could not create a session", fatal)
            connectionStatus.value = ConnectionStatus.Disconnected("ARCore could not start a session: ${fatal.message ?: "FatalException"}")
        }
    }

    private fun connect(host: String, port: Int) {
        connection?.stop()
        connection = LaptopConnection(host, port) { status -> connectionStatus.value = status }.also { it.start() }
    }

    private fun renderer(): DepthCaptureRenderer = DepthCaptureRenderer(
        sessionProvider = { session },
        onFrame = { message -> connection?.offer(message) },
        onState = { state -> captureState.value = state },
    )

    @Composable
    private fun Screen(
        captureState: MutableStateFlow<CaptureState>,
        connectionStatus: MutableStateFlow<ConnectionStatus>,
        onConnect: (String, Int) -> Unit,
        glSurface: (GLSurfaceView) -> Unit,
    ) {
        val capture by captureState.collectAsState()
        val status by connectionStatus.collectAsState()
        var host by remember { mutableStateOf(startupHost) }
        var port by remember { mutableStateOf(startupPort.toString()) }

        Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            Text("Depth to the laptop", style = MaterialTheme.typography.titleLarge)
            Row(modifier = Modifier.fillMaxWidth().padding(top = 12.dp)) {
                OutlinedTextField(value = host, onValueChange = { host = it }, label = { Text("laptop address") }, modifier = Modifier.weight(2f))
                OutlinedTextField(value = port, onValueChange = { port = it }, label = { Text("port") }, modifier = Modifier.weight(1f).padding(start = 8.dp))
            }
            Button(onClick = { port.toIntOrNull()?.let { onConnect(host, it) } }, modifier = Modifier.padding(top = 8.dp)) {
                Text("Connect")
            }
            Text(describe(status), modifier = Modifier.padding(top = 12.dp))
            Text(describe(capture), modifier = Modifier.padding(top = 4.dp))
            // The GL surface is what drives ARCore. It draws black. The two lines above are the
            // display, and the arrow is on the laptop's web page in the browser.
            AndroidView(
                factory = { context ->
                    GLSurfaceView(context).apply {
                        setEGLContextClientVersion(2)
                        setRenderer(renderer())
                        renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
                        glSurface(this)
                    }
                },
                modifier = Modifier.fillMaxWidth().weight(1f).padding(top = 12.dp),
            )
        }
    }

    private fun describe(status: ConnectionStatus): String = when (status) {
        is ConnectionStatus.Connecting -> "Connecting to ${status.host}:${status.port}."
        is ConnectionStatus.Connected -> "Connected to ${status.host}:${status.port}. Sent ${status.framesSent}, dropped ${status.framesDropped} as stale."
        is ConnectionStatus.Disconnected -> "Not connected: ${status.reason}."
    }

    private fun describe(state: CaptureState): String = when (state) {
        CaptureState.CameraUnavailable -> "Camera not available yet."
        is CaptureState.Running -> buildString {
            append(if (state.tracking) "Tracking. " else "Not tracking, move the phone. ")
            append("${state.framesWithDepth} of ${state.frames} frames had depth")
            state.depthSize?.let { append(" at $it") }
            append(if (state.hasFloor) ". Floor found." else ". No floor yet.")
        }
    }

    private companion object {
        const val TAG = "MainActivity"
        const val EXTRA_HOST = "host"
        const val EXTRA_PORT = "port"
        // The laptop on a phone hotspot is usually the first client. Edit on screen when not.
        const val DEFAULT_HOST = "192.168.43.1"
        const val DEFAULT_PORT = 9000
    }
}
