package com.example.neuromorphicpaths

import android.Manifest.permission.BLUETOOTH
import android.Manifest.permission.BLUETOOTH_CONNECT
import android.Manifest.permission.ACCESS_COARSE_LOCATION
import android.Manifest.permission.ACCESS_FINE_LOCATION
import android.Manifest.permission.INTERNET
import android.Manifest.permission.RECORD_AUDIO
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts.RequestMultiplePermissions
import androidx.activity.result.contract.ActivityResultContracts.RequestPermission
import androidx.activity.viewModels
import androidx.core.content.ContextCompat
import com.example.neuromorphicpaths.ui.theme.DisplayAccessTheme
import com.example.neuromorphicpaths.wearables.WearablesViewModel
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.externalsampleapps.displayaccess.ui.AppScaffold
import kotlin.coroutines.resume
import kotlinx.coroutines.CancellableContinuation
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class MainActivity : ComponentActivity() {
    companion object {
        // Required Android permissions for the DAT SDK to function properly
        val PERMISSIONS: Array<String> =
            arrayOf(
                BLUETOOTH,
                BLUETOOTH_CONNECT,
                INTERNET,
                ACCESS_COARSE_LOCATION,
                ACCESS_FINE_LOCATION,
            )
    }

    val viewModel: WearablesViewModel by viewModels()

    private val permissionCheckLauncher =
        registerForActivityResult(RequestMultiplePermissions()) { permissionsResult ->
            viewModel.onPermissionsResult(permissionsResult) {
                // Initialize the DAT SDK once the permissions are granted
                // This is REQUIRED before using any Wearables APIs
                Wearables.initialize(this)
            }
        }

    private var permissionContinuation: CancellableContinuation<PermissionStatus>? = null
    private val permissionMutex = Mutex()
    // Requesting wearable device permissions via the Meta AI app
    private val permissionsResultLauncher =
        registerForActivityResult(Wearables.RequestPermissionContract()) { result ->
            val permissionStatus = result.getOrDefault(PermissionStatus.Denied)
            permissionContinuation?.resume(permissionStatus)
            permissionContinuation = null
        }

    // Convenience method to make a permission request in a sequential manner
    // Uses a Mutex to ensure requests are processed one at a time, preventing race conditions
    suspend fun requestWearablesPermission(permission: Permission): PermissionStatus {
        return permissionMutex.withLock {
            suspendCancellableCoroutine { continuation ->
                permissionContinuation = continuation
                continuation.invokeOnCancellation { permissionContinuation = null }
                permissionsResultLauncher.launch(permission)
            }
        }
    }

    private var audioPermissionContinuation: CancellableContinuation<Boolean>? = null
    // Phone microphone permission, requested in context when recording with sound-in-video on.
    private val recordAudioPermissionLauncher =
        registerForActivityResult(RequestPermission()) { granted ->
            audioPermissionContinuation?.resume(granted)
            audioPermissionContinuation = null
        }

    // Requests RECORD_AUDIO for sound-in-video. Returns true if granted (already or just now); false
    // if denied, so recording can proceed video-only instead of being blocked.
    suspend fun requestRecordAudioPermission(): Boolean {
        if (
            ContextCompat.checkSelfPermission(this, RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        ) {
            return true
        }
        return permissionMutex.withLock {
            suspendCancellableCoroutine { continuation ->
                audioPermissionContinuation = continuation
                continuation.invokeOnCancellation { audioPermissionContinuation = null }
                recordAudioPermissionLauncher.launch(RECORD_AUDIO)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DisplayAccessTheme {
                AppScaffold(
                    wearablesViewModel = viewModel,
                    onRequestWearablesPermission = ::requestWearablesPermission,
                    onRequestRecordAudioPermission = ::requestRecordAudioPermission,
                )
            }
        }
    }

    override fun onStart() {
        super.onStart()
        // First, ensure the app has necessary Android permissions
        permissionCheckLauncher.launch(PERMISSIONS)
    }
}
