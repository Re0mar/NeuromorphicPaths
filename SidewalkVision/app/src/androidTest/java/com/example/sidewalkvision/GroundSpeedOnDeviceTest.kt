package com.example.sidewalkvision

import android.content.Context
import android.location.Location
import android.location.LocationManager
import android.location.provider.ProviderProperties
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

private const val GPS = LocationManager.GPS_PROVIDER
private const val METERS_PER_DEGREE_LATITUDE = 111_320.0

// A walk north across the Radboud campus at walking pace, one fix a second like a phone's GPS.
private const val START_LATITUDE = 51.8195
private const val START_LONGITUDE = 5.8650
private const val WALKING_SPEED = 1.4
private const val FIX_INTERVAL_MS = 1_000L
private const val FIX_COUNT = 5

@RunWith(AndroidJUnit4::class)
class GroundSpeedOnDeviceTest {
    private lateinit var context: Context
    private lateinit var locationManager: LocationManager
    private lateinit var tracker: GroundSpeedTracker
    // Android throttles location updates to a background app to a few an hour. An activity on
    // screen keeps the app in the foreground, as it is when someone uses the camera screen.
    private lateinit var foreground: ActivityScenario<MainActivity>

    // Reads the command's output to the end, which is how to wait for it to finish. Without that
    // the permission and mock-location settings may not have applied when the test goes on.
    private fun runShell(command: String) {
        val output = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand(command)
        ParcelFileDescriptor.AutoCloseInputStream(output).use { it.readBytes() }
    }

    // Location updates arrive on the main thread some time after a fix is injected.
    private fun awaitSpeed(): Double? {
        val deadline = SystemClock.elapsedRealtime() + FIX_INTERVAL_MS * 3
        while (tracker.metersPerSecond.value == null && SystemClock.elapsedRealtime() < deadline) {
            Thread.sleep(50)
        }
        return tracker.metersPerSecond.value
    }

    @Before
    fun startTestProvider() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        runShell("pm grant ${context.packageName} android.permission.ACCESS_FINE_LOCATION")
        runShell("pm grant ${context.packageName} android.permission.ACCESS_COARSE_LOCATION")
        // Test providers are refused unless this app is the selected mock location app.
        runShell("appops set ${context.packageName} android:mock_location allow")

        foreground = ActivityScenario.launch(MainActivity::class.java)
        locationManager = context.getSystemService(LocationManager::class.java)
        locationManager.addTestProvider(
            GPS,
            ProviderProperties.Builder()
                .setAccuracy(ProviderProperties.ACCURACY_FINE)
                .setPowerUsage(ProviderProperties.POWER_USAGE_LOW)
                .build()
        )
        locationManager.setTestProviderEnabled(GPS, true)
        tracker = GroundSpeedTracker(context)
        assertTrue("tracker should start with location permission", tracker.start())
    }

    @After
    fun stopTestProvider() {
        tracker.stop()
        locationManager.removeTestProvider(GPS)
        runShell("appops set ${context.packageName} android:mock_location default")
        foreground.close()
    }

    private fun walk(reportSpeed: Boolean) {
        var latitude = START_LATITUDE
        repeat(FIX_COUNT) {
            val fix = Location(GPS).apply {
                this.latitude = latitude
                longitude = START_LONGITUDE
                altitude = 10.0
                accuracy = 3f
                time = System.currentTimeMillis()
                elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos()
                if (reportSpeed) {
                    speed = WALKING_SPEED.toFloat()
                    speedAccuracyMetersPerSecond = 0.2f
                }
            }
            locationManager.setTestProviderLocation(GPS, fix)
            latitude += WALKING_SPEED * (FIX_INTERVAL_MS / 1_000.0) / METERS_PER_DEGREE_LATITUDE
            Thread.sleep(FIX_INTERVAL_MS)
        }
    }

    @Test
    fun theReceiversOwnSpeedIsShownWhenReported() {
        walk(reportSpeed = true)

        val speed = awaitSpeed()
        assertNotNull(speed)
        assertEquals(WALKING_SPEED, speed!!, 0.05)
    }

    @Test
    fun distanceOverTimeIsUsedWhenNoSpeedIsReported() {
        walk(reportSpeed = false)

        // The first fix has nothing to compare with, so later readings carry the speed. The fixes
        // are timed by the device clock, which sleep rounding moves by a few percent.
        val speed = awaitSpeed()
        assertNotNull(speed)
        assertEquals(WALKING_SPEED, speed!!, 0.15)
    }
}
