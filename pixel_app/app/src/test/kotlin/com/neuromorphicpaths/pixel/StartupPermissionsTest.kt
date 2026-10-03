package com.neuromorphicpaths.pixel

import android.Manifest
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * What the app asks for at startup and what it does with a refusal. The dialog itself is device
 * only. What can silently break, a permission dropped from the request or the manifest, is here.
 */
class StartupPermissionsTest {
    private fun grantedExcept(vararg refused: String): (String) -> Boolean = { it !in refused }

    @Test
    fun theStartupRequestAsksForTheLocalNetworkAndTheCamera() {
        assertTrue(Manifest.permission.ACCESS_LOCAL_NETWORK in StartupPermissions.requested)
        assertTrue(Manifest.permission.CAMERA in StartupPermissions.requested)
    }

    @Test
    fun theManifestDeclaresEveryPermissionTheAppRequests() {
        // Android refuses an undeclared permission without showing a dialog, so a request missing
        // from the manifest fails as silently as no request at all. Unit tests run from the module.
        val manifest = File("src/main/AndroidManifest.xml").readText()

        for (permission in StartupPermissions.requested) {
            assertTrue("android:name=\"$permission\"" in manifest, "$permission is requested but not declared")
        }
    }

    @Test
    fun everythingGrantedCapturesWithNoWarning() {
        val outcome = StartupPermissions.outcome(grantedExcept())

        assertTrue(outcome.canCapture)
        assertNull(outcome.warning)
    }

    @Test
    fun aRefusedLocalNetworkStillCapturesButSaysWhyTheLaptopIsUnreachable() {
        val outcome = StartupPermissions.outcome(grantedExcept(Manifest.permission.ACCESS_LOCAL_NETWORK))

        assertTrue(outcome.canCapture, "a laptop on a public address is still reachable, so capture goes on")
        assertEquals(StartupPermissions.LOCAL_NETWORK_REFUSED, outcome.warning)
    }

    @Test
    fun aRefusedCameraCannotCapture() {
        val outcome = StartupPermissions.outcome(grantedExcept(Manifest.permission.CAMERA))

        assertFalse(outcome.canCapture)
        assertEquals(StartupPermissions.CAMERA_REFUSED, outcome.warning)
    }

    @Test
    fun bothRefusedReportsTheCameraWhichStopsEverything() {
        val outcome = StartupPermissions.outcome(grantedExcept(Manifest.permission.CAMERA, Manifest.permission.ACCESS_LOCAL_NETWORK))

        assertFalse(outcome.canCapture)
        assertEquals(StartupPermissions.CAMERA_REFUSED, outcome.warning)
    }
}
