package com.neuromorphicpaths.pixel

import android.content.pm.PackageManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
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
import com.neuromorphicpaths.pixel.ui.ArrowOverlay
import com.neuromorphicpaths.pixel.wire.ConnectionStatus
import com.neuromorphicpaths.pixel.wire.LaptopConnection
import com.neuromorphicpaths.pixel.wire.PathConnection
import com.neuromorphicpaths.pixel.wire.PathConnectionStatus
import com.neuromorphicpaths.pixel.wire.ReceivedPath
import com.neuromorphicpaths.pixel.wire.RecordingProgress
import com.neuromorphicpaths.pixel.wire.ReplayStatus
import com.neuromorphicpaths.pixel.wire.WalkRecorder
import com.neuromorphicpaths.pixel.wire.WalkReplayer
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlin.concurrent.thread
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Sends the Pixel's ARCore depth to the laptop pipeline and receives the paths it plans. One
 * screen, three lines that matter: whether depth frames are flowing, whether the laptop is
 * receiving them, and whether paths are coming back.
 *
 * It can also record a walk with nothing connected and replay the recording to the laptop later,
 * for walks where the phone can't reach the laptop. The laptop records a replay like a live walk.
 */
class MainActivity : ComponentActivity() {
    private var session: Session? = null
    private var connection: LaptopConnection? = null
    private var pathConnection: PathConnection? = null
    private var surface: GLSurfaceView? = null
    private val captureState = MutableStateFlow<CaptureState>(CaptureState.CameraUnavailable)
    private val connectionStatus = MutableStateFlow<ConnectionStatus>(ConnectionStatus.Disconnected("not started"))
    private val pathStatus = MutableStateFlow<PathConnectionStatus>(PathConnectionStatus.Disconnected("not started"))
    private val latestPath = MutableStateFlow<ReceivedPath?>(null)
    // Read on the GL thread for every frame, written on the UI thread.
    @Volatile private var recorder: WalkRecorder? = null
    private var replayer: WalkReplayer? = null
    private val recordingStatus = MutableStateFlow<RecordingStatus>(RecordingStatus.Idle)
    private val replayStatus = MutableStateFlow<ReplayStatus?>(null)
    private var installRequested = false
    private var startupHost = DEFAULT_HOST
    private var startupPort = DEFAULT_PORT
    private var startupPathPort = DEFAULT_PATH_PORT

    private val permissionWarning = MutableStateFlow<String?>(null)

    // The result map is ignored on purpose. applyPermissions asks the system again, so a permission
    // granted on an earlier launch and left out of this request still counts.
    private val startupPermissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        applyPermissions()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Started from adb with --es host and --ei port, the app connects by itself. That is how
        // the emulator run drives it with nobody typing into a headless screen.
        val hostExtra = intent.getStringExtra(EXTRA_HOST)
        if (hostExtra != null) {
            startupHost = hostExtra
            startupPort = intent.getIntExtra(EXTRA_PORT, DEFAULT_PORT)
            startupPathPort = intent.getIntExtra(EXTRA_PATH_PORT, DEFAULT_PATH_PORT)
            connect(startupHost, startupPort, startupPathPort)
        }
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    Screen(
                        captureState = captureState,
                        connectionStatus = connectionStatus,
                        pathStatus = pathStatus,
                        latestPath = latestPath,
                        permissionWarning = permissionWarning,
                        onConnect = { host, port, pathPort -> connect(host, port, pathPort) },
                        glSurface = { view -> surface = view },
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        val missing = StartupPermissions.requested.filterNot(::isGranted)
        if (missing.isEmpty()) applyPermissions() else startupPermissions.launch(missing.toTypedArray())
        surface?.onResume()
    }

    private fun isGranted(permission: String): Boolean =
        ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED

    private fun applyPermissions() {
        val outcome = StartupPermissions.outcome(::isGranted)
        permissionWarning.value = outcome.warning
        if (outcome.canCapture) startSession()
    }

    override fun onPause() {
        super.onPause()
        surface?.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        connection?.stop()
        pathConnection?.stop()
        replayer?.stop()
        // Writes out what is still queued, so a walk recorded right up to closing the app keeps its end.
        recorder?.stop()
        recorder = null
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

    /** Both connections go to the one laptop address. The phone opens both, the laptop listens on both. */
    private fun connect(host: String, port: Int, pathPort: Int) {
        // The laptop takes one depth connection at a time, so a live connection ends any replay.
        replayer?.stop()
        replayer = null
        connection?.stop()
        pathConnection?.stop()
        connection = LaptopConnection(host, port) { status -> connectionStatus.value = status }.also { it.start() }
        pathConnection = PathConnection(
            host = host,
            port = pathPort,
            onStatus = { status -> pathStatus.value = status },
            onPath = { path -> latestPath.value = path },
        ).also { it.start() }
    }

    private fun renderer(): DepthCaptureRenderer = DepthCaptureRenderer(
        sessionProvider = { session },
        onFrame = { message ->
            connection?.offer(message)
            recorder?.offer(message)
        },
        onState = { state -> captureState.value = state },
    )

    private fun startRecording() {
        if (recorder != null) return
        val directory = getExternalFilesDir(WALKS_DIRECTORY)
        if (directory == null) {
            recordingStatus.value = RecordingStatus.Unavailable("the app has no storage to record to right now")
            return
        }
        val name = "walk_${LocalDateTime.now().format(FILE_TIME_FORMAT)}$WALK_SUFFIX"
        val output = try {
            FileOutputStream(File(directory, name))
        } catch (unwritable: IOException) {
            recordingStatus.value = RecordingStatus.Unavailable("can't create $name: ${unwritable.message}")
            return
        }
        recordingStatus.value = RecordingStatus.Recording(name, RecordingProgress(0, 0, 0, null))
        recorder = WalkRecorder(output, onProgress = { progress -> recordingStatus.value = RecordingStatus.Recording(name, progress) })
            .also { it.start() }
    }

    private fun stopRecording() {
        val stopping = recorder ?: return
        // Stop handing it frames first, then let it write out what is queued, off the UI thread.
        recorder = null
        val name = (recordingStatus.value as? RecordingStatus.Recording)?.name ?: "the recording"
        thread(name = "walk-recorder-stop", isDaemon = true) {
            stopping.stop()
            recordingStatus.value = RecordingStatus.Saved(name, stopping.progress)
        }
    }

    /** Sends the newest recording to the laptop at host and port, in place of the live connection. */
    private fun replayLatest(host: String, port: Int) {
        if (recorder != null) {
            replayStatus.value = ReplayStatus.Failed("stop the recording before replaying", 0)
            return
        }
        val latest = getExternalFilesDir(WALKS_DIRECTORY)
            ?.listFiles { file -> file.name.endsWith(WALK_SUFFIX) }
            ?.maxByOrNull { it.lastModified() }
        if (latest == null) {
            replayStatus.value = ReplayStatus.Failed("no recorded walk on this phone", 0)
            return
        }
        // The laptop takes one depth connection at a time, so the live one makes way.
        connection?.stop()
        connection = null
        connectionStatus.value = ConnectionStatus.Disconnected("replaying ${latest.name} instead")
        replayer?.stop()
        replayer = WalkReplayer(host, port, { FileInputStream(latest) }, { status -> replayStatus.value = status }).also { it.start() }
    }

    @Composable
    private fun Screen(
        captureState: MutableStateFlow<CaptureState>,
        connectionStatus: MutableStateFlow<ConnectionStatus>,
        pathStatus: MutableStateFlow<PathConnectionStatus>,
        latestPath: MutableStateFlow<ReceivedPath?>,
        permissionWarning: MutableStateFlow<String?>,
        onConnect: (String, Int, Int) -> Unit,
        glSurface: (GLSurfaceView) -> Unit,
    ) {
        val warning by permissionWarning.collectAsState()
        val capture by captureState.collectAsState()
        val status by connectionStatus.collectAsState()
        val paths by pathStatus.collectAsState()
        val path by latestPath.collectAsState()
        val recording by recordingStatus.collectAsState()
        val replay by replayStatus.collectAsState()
        var host by remember { mutableStateOf(startupHost) }
        var port by remember { mutableStateOf(startupPort.toString()) }
        var pathPort by remember { mutableStateOf(startupPathPort.toString()) }

        Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
            Text("Depth to the laptop", style = MaterialTheme.typography.titleLarge)
            Row(modifier = Modifier.fillMaxWidth().padding(top = 12.dp)) {
                OutlinedTextField(value = host, onValueChange = { host = it }, label = { Text("laptop address") }, modifier = Modifier.weight(2f))
                OutlinedTextField(value = port, onValueChange = { port = it }, label = { Text("depth port") }, modifier = Modifier.weight(1f).padding(start = 8.dp))
                OutlinedTextField(value = pathPort, onValueChange = { pathPort = it }, label = { Text("path port") }, modifier = Modifier.weight(1f).padding(start = 8.dp))
            }
            Button(
                onClick = {
                    val depthPort = port.toIntOrNull()
                    val pathsPort = pathPort.toIntOrNull()
                    if (depthPort != null && pathsPort != null) onConnect(host, depthPort, pathsPort)
                },
                modifier = Modifier.padding(top = 8.dp),
            ) {
                Text("Connect")
            }
            Row(modifier = Modifier.padding(top = 8.dp)) {
                if (recording is RecordingStatus.Recording) {
                    Button(onClick = { stopRecording() }) { Text("Stop recording") }
                } else {
                    Button(onClick = { startRecording() }) { Text("Record walk") }
                }
                Button(
                    onClick = { port.toIntOrNull()?.let { depthPort -> replayLatest(host, depthPort) } },
                    modifier = Modifier.padding(start = 8.dp),
                ) {
                    Text("Replay latest walk")
                }
            }
            Text(describe(recording), modifier = Modifier.padding(top = 8.dp))
            replay?.let { Text(describe(it), modifier = Modifier.padding(top = 4.dp)) }
            // Its own line, because the connection status is rewritten on every attempt and would
            // bury the one thing that explains why the attempts keep failing.
            warning?.let { Text("Permission: $it.", color = MaterialTheme.colorScheme.error, modifier = Modifier.padding(top = 12.dp)) }
            Text(describe(status), modifier = Modifier.padding(top = 12.dp))
            Text(describe(capture), modifier = Modifier.padding(top = 4.dp))
            Text(describe(paths), modifier = Modifier.padding(top = 4.dp))
            // The GL surface drives ARCore and draws the camera. The arrow is Compose over it,
            // which keeps display code out of the renderer and makes its arithmetic testable.
            Box(modifier = Modifier.fillMaxWidth().weight(1f).padding(top = 12.dp)) {
                AndroidView(
                    factory = { context ->
                        GLSurfaceView(context).apply {
                            setEGLContextClientVersion(2)
                            setRenderer(renderer())
                            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
                            glSurface(this)
                        }
                    },
                    modifier = Modifier.fillMaxSize(),
                )
                // The same clock the connection stamps paths with, so the age is one clock's difference.
                ArrowOverlay(received = path, nowMillis = { SystemClock.elapsedRealtime() }, modifier = Modifier.fillMaxSize())
            }
        }
    }

    private fun describe(status: ConnectionStatus): String = when (status) {
        is ConnectionStatus.Connecting -> "Connecting to ${status.host}:${status.port}."
        is ConnectionStatus.Connected -> "Connected to ${status.host}:${status.port}. Sent ${status.framesSent}, dropped ${status.framesDropped} as stale."
        is ConnectionStatus.Disconnected -> "Not connected: ${status.reason}."
    }

    private fun describe(status: PathConnectionStatus): String = when (status) {
        is PathConnectionStatus.Connecting -> "Paths: connecting to ${status.host}:${status.port}."
        is PathConnectionStatus.Connected -> buildString {
            append("Paths: connected, ${status.received} received, ${status.refused} refused.")
            status.lastRefusal?.let { append(" Last refusal: $it") }
        }
        is PathConnectionStatus.Disconnected -> "Paths: not connected: ${status.reason}."
    }

    private fun describe(status: RecordingStatus): String = when (status) {
        RecordingStatus.Idle -> "Not recording."
        is RecordingStatus.Recording -> "Recording ${status.name}: ${describe(status.progress)}"
        is RecordingStatus.Saved -> "Saved ${status.name}: ${describe(status.progress)}"
        is RecordingStatus.Unavailable -> "Can't record: ${status.reason}."
    }

    private fun describe(progress: RecordingProgress): String = buildString {
        append("${progress.framesWritten} frames, ${progress.framesDropped} dropped, ")
        append(String.format(Locale.ROOT, "%.1f MB.", progress.bytesWritten / BYTES_PER_MEGABYTE))
        progress.failure?.let { append(" Writing stopped: $it.") }
    }

    private fun describe(status: ReplayStatus): String = when (status) {
        is ReplayStatus.Connecting -> "Replay: connecting to ${status.host}:${status.port}."
        is ReplayStatus.Replaying -> "Replay: ${status.framesSent} frames sent to ${status.host}:${status.port}."
        is ReplayStatus.Finished -> buildString {
            append("Replay finished: ${status.framesSent} frames sent.")
            if (status.truncatedBytes > 0) append(" The recording's last ${status.truncatedBytes} bytes were a cut-off frame and were left out.")
        }
        is ReplayStatus.Failed -> "Replay stopped after ${status.framesSent} frames: ${status.reason}. Replay into a fresh laptop directory."
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
        const val EXTRA_PATH_PORT = "path_port"
        // The laptop on a phone hotspot is usually the first client. Edit on screen when not.
        const val DEFAULT_HOST = "192.168.43.1"
        const val DEFAULT_PORT = 9000
        // The laptop's --phone-port default. Its own number, not derived from the depth port.
        const val DEFAULT_PATH_PORT = 9100
        // Under the app's own storage, which needs no permission and which adb pull can reach.
        const val WALKS_DIRECTORY = "walks"
        const val WALK_SUFFIX = ".bin"
        val FILE_TIME_FORMAT: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss", Locale.ROOT)
        const val BYTES_PER_MEGABYTE = 1_000_000.0
    }
}

/** What the screen shows about recording a walk. */
private sealed interface RecordingStatus {
    data object Idle : RecordingStatus

    data class Recording(val name: String, val progress: RecordingProgress) : RecordingStatus

    data class Saved(val name: String, val progress: RecordingProgress) : RecordingStatus

    data class Unavailable(val reason: String) : RecordingStatus
}
