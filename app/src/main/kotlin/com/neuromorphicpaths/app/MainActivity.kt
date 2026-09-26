package com.neuromorphicpaths.app

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.net.Uri
import android.os.Bundle
import android.os.IBinder
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import com.neuromorphicpaths.app.ui.MainScreen
import com.neuromorphicpaths.input.Recording
import com.neuromorphicpaths.output.FloatingForm
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File

/**
 * The controls. Binds to [GuidanceService], which runs the pipeline, and draws that service's
 * screen display while it is in front. It asks for the permissions the service needs, since
 * only a visible screen can, and it tells the floating overlay to stay out of the way while
 * this screen is showing.
 */
class MainActivity : ComponentActivity() {

    private val service = MutableStateFlow<GuidanceService?>(null)

    // The adb extras wait here until the service is bound, then run once.
    private var pendingLaunchExtras: Bundle? = null

    // A floating form picked before the draw-over-apps permission was granted, applied on return from settings.
    private var formAwaitingPermission: FloatingForm? = null

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            val bound = (binder as GuidanceService.LocalBinder).service
            bound.floatingDisplay.setAppVisible(true)
            service.value = bound
            pendingLaunchExtras?.let { applyLaunchExtras(bound, it) }
            pendingLaunchExtras = null
        }

        override fun onServiceDisconnected(name: ComponentName) {
            service.value = null
        }
    }

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            val bound = service.value ?: return@registerForActivityResult
            // Location is optional. Without it the walker's speed stays unknown and the field uses its default.
            if (granted[Manifest.permission.ACCESS_FINE_LOCATION] == true) bound.startGroundSpeed()
            // Notifications are optional too. Without them the service still runs, its notice is just hidden.
            if (granted[Manifest.permission.CAMERA] == true) bound.startCamera()
        }

    private val pickRecording =
        registerForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
            val recording = uri?.let { Recording.fromTree(this, it) }
            val bound = service.value
            when {
                recording == null -> Log.w(TAG, "No video in the picked folder")
                bound == null -> Log.w(TAG, "Service not bound, recording not started")
                else -> bound.startRecording(recording)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // adb hooks, applied once the service is bound:
        //   adb shell am start -n com.neuromorphicpaths/.app.MainActivity --es recording outdoor1
        //   --ez segmenter false      --ez xnnpack true      --es overlay corner|full|off
        if (savedInstanceState == null) pendingLaunchExtras = intent.extras
        setContent {
            val bound by service.collectAsState()
            MaterialTheme {
                val current = bound
                if (current == null) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) { Text("Starting") }
                } else {
                    val choice by current.detectorChoice.collectAsState()
                    val surfaces by current.segmenterOn.collectAsState()
                    val running by current.runningSource.collectAsState()
                    val form by current.floatingDisplay.form.collectAsState()
                    MainScreen(
                        display = current.screenDisplay,
                        detectorChoice = choice,
                        yoloWorldAvailable = current.yoloWorldAvailable,
                        segmenterOn = surfaces,
                        segmenterAvailable = current.segmenterAvailable,
                        running = running != null,
                        floatingForm = form,
                        onDetectorChoiceChange = current::setDetectorChoice,
                        onSegmenterChange = current::setSegmenterOn,
                        onStartCamera = {
                            requestPermissions.launch(
                                arrayOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.POST_NOTIFICATIONS),
                            )
                        },
                        onOpenRecording = { pickRecording.launch(null) },
                        onStop = current::stopGuidance,
                        onFloatingFormChange = { picked -> pickFloatingForm(current, picked) },
                    )
                }
            }
        }
    }

    override fun onStart() {
        super.onStart()
        bindService(Intent(this, GuidanceService::class.java), connection, Context.BIND_AUTO_CREATE)
    }

    override fun onResume() {
        super.onResume()
        val waiting = formAwaitingPermission ?: return
        formAwaitingPermission = null
        if (Settings.canDrawOverlays(this)) service.value?.floatingDisplay?.setForm(waiting)
    }

    override fun onStop() {
        // From here on the floating overlay is the only thing showing the guidance.
        service.value?.floatingDisplay?.setAppVisible(false)
        unbindService(connection)
        service.value = null
        super.onStop()
    }

    private fun pickFloatingForm(bound: GuidanceService, form: FloatingForm) {
        if (form != FloatingForm.OFF && !Settings.canDrawOverlays(this)) {
            // Drawing over other apps is a special permission, granted on its own settings page.
            formAwaitingPermission = form
            startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")))
            return
        }
        bound.floatingDisplay.setForm(form)
    }

    private fun applyLaunchExtras(bound: GuidanceService, extras: Bundle) {
        if (extras.containsKey(EXTRA_SEGMENTER)) bound.setSegmenterOn(extras.getBoolean(EXTRA_SEGMENTER))
        bound.segmenterUsesXnnpack = extras.getBoolean(EXTRA_XNNPACK, false)
        extras.getString(EXTRA_OVERLAY)?.let { name ->
            val form = OVERLAY_NAMES[name]
            if (form != null) bound.floatingDisplay.setForm(form) else Log.w(TAG, "Unknown overlay $name, expected one of ${OVERLAY_NAMES.keys}")
        }
        extras.getString(EXTRA_RECORDING)?.let { name ->
            val recording = Recording.fromDirectory(this, File(filesDir, "$RECORDINGS_DIR/$name"))
            if (recording != null) bound.startRecording(recording) else Log.w(TAG, "No recording named $name under $RECORDINGS_DIR")
        }
    }

    private companion object {
        const val TAG = "MainActivity"
        const val EXTRA_RECORDING = "recording"
        const val EXTRA_SEGMENTER = "segmenter"
        const val EXTRA_XNNPACK = "xnnpack"
        const val EXTRA_OVERLAY = "overlay"
        const val RECORDINGS_DIR = "recordings"
        val OVERLAY_NAMES = mapOf("off" to FloatingForm.OFF, "corner" to FloatingForm.CORNER, "full" to FloatingForm.FULL_SCREEN)
    }
}
