package com.neuromorphicpaths.pixel

import android.Manifest

/**
 * The runtime permissions the app asks for when it starts, and what each refusal costs.
 *
 * Android 17 drops an app's connections to private addresses such as 10.x and 192.168.x unless the
 * app holds the local network permission. A laptop on home or lab Wi-Fi sits at one of those, so
 * without the permission both connections time out and nothing on either side says why. A laptop
 * on a public address, as on eduroam, is reached without it, so a refusal is a warning, not a stop.
 */
internal object StartupPermissions {
    const val CAMERA_REFUSED = "camera permission refused"
    const val LOCAL_NETWORK_REFUSED =
        "local network permission refused, so a laptop on a private address such as 10.x or 192.168.x cannot be reached"

    /** Everything asked for at startup. Each one must also be declared in the manifest, or Android refuses it without asking. */
    val requested: List<String> = listOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_LOCAL_NETWORK)

    /**
     * What the app can do with the permissions it holds.
     *
     * @param isGranted whether the system currently grants a permission, asked fresh rather than
     *     read from a dialog's result, so one granted on an earlier launch counts too
     */
    fun outcome(isGranted: (String) -> Boolean): PermissionOutcome {
        val canCapture = isGranted(Manifest.permission.CAMERA)
        val warning = when {
            !canCapture -> CAMERA_REFUSED
            !isGranted(Manifest.permission.ACCESS_LOCAL_NETWORK) -> LOCAL_NETWORK_REFUSED
            else -> null
        }
        return PermissionOutcome(canCapture = canCapture, warning = warning)
    }
}

/**
 * @property canCapture the camera is granted, so the ARCore session can start
 * @property warning what to show the user about a refusal, or null when nothing was refused
 */
internal data class PermissionOutcome(val canCapture: Boolean, val warning: String?)
