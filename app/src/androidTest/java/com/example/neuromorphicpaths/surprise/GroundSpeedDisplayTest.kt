// GroundSpeedDisplayTest - GPS ground speed, from injected fixes to the camera screen chip.
//
// Replaces the GPS provider with a test provider and feeds it a synthetic walk: a fix every 1 to
// 2.5 seconds, moving due north at a known speed per step. Each fix carries its own timestamp, so
// the test controls both the distance and the time that Surprise divides.
//
// The number of fixes defaults to 60 (about two minutes). Override it for a longer soak with
//   -Pandroid.testInstrumentationRunnerArguments.speedSamples=600

package com.example.neuromorphicpaths.surprise

import android.Manifest
import android.content.Context
import android.location.Location
import android.location.LocationManager
import android.location.provider.ProviderProperties
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import androidx.compose.ui.semantics.SemanticsProperties
import androidx.compose.ui.semantics.getOrNull
import androidx.compose.ui.test.ExperimentalTestApi
import androidx.compose.ui.test.hasTestTag
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.isEnabled
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import com.example.neuromorphicpaths.MainActivity
import com.example.neuromorphicpaths.R
import com.meta.wearable.dat.core.types.Permission
import com.meta.wearable.dat.core.types.PermissionStatus
import com.meta.wearable.dat.mockdevice.MockDeviceKit
import com.meta.wearable.dat.mockdevice.api.GlassesModel
import com.meta.wearable.dat.mockdevice.api.MockDeviceKitConfig
import java.io.File
import java.io.FileOutputStream
import kotlin.math.abs
import kotlin.math.max
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@OptIn(ExperimentalTestApi::class)
@RunWith(AndroidJUnit4::class)
@LargeTest
class GroundSpeedDisplayTest {

  companion object {
    private const val TAG = "GroundSpeedDisplayTest"
    private const val GPS = LocationManager.GPS_PROVIDER
    private const val DEFAULT_SAMPLE_COUNT = 60
    private const val STREAM_TIMEOUT_MS = 20_000L
    // The view model copies the speed into the UI once a second, so allow a few ticks.
    private const val SPEED_TIMEOUT_MS = 4_000L
    private const val POLL_INTERVAL_MS = 100L

    // Radboud campus, so a failing run's coordinates are recognizable.
    private const val START_LATITUDE = 51.8193
    private const val START_LONGITUDE = 5.8570

    // Length of one degree of latitude on a sphere of radius 6,371 km. Location.distanceTo uses
    // the WGS84 ellipsoid instead, which reads about 0.06% longer at this latitude. Deliberately
    // independent, so the test isn't checking distanceTo against itself.
    private const val METERS_PER_DEGREE_LATITUDE = 111_195.0

    // Standing, strolling, brisk walking and one jog. The two lists have coprime lengths (7 and
    // 5), so speed and interval pairings only repeat after 35 fixes. Neighboring speeds always
    // differ, so every step moves the display and a dropped fix can't pass by accident.
    private val SPEEDS_METERS_PER_SECOND = listOf(0.0, 1.4, 0.8, 1.6, 2.0, 3.2, 1.1)
    // Nothing under 1 s. Surprise requests updates at 1 s and the platform drops fixes that
    // arrive faster than the requested interval.
    private val INTERVALS_MS = listOf(1_000L, 1_500L, 2_000L, 1_000L, 2_500L)

    private val SPEED_TEXT_PATTERN = Regex("""(-?\d+\.\d+) m/s""")
  }

  // Granted before launch. MainActivity requests these in onStart and would otherwise show the
  // system dialog over the screen under test.
  @get:Rule(order = 0)
  val permissionRule: GrantPermissionRule =
      GrantPermissionRule.grant(
          Manifest.permission.ACCESS_FINE_LOCATION,
          Manifest.permission.ACCESS_COARSE_LOCATION,
          Manifest.permission.BLUETOOTH_CONNECT,
          Manifest.permission.CAMERA,
          Manifest.permission.RECORD_AUDIO,
      )

  @get:Rule(order = 1) val composeTestRule = createAndroidComposeRule<MainActivity>()

  private val targetContext: Context
    get() = InstrumentationRegistry.getInstrumentation().targetContext.applicationContext

  private val locationManager: LocationManager
    get() = targetContext.getSystemService(LocationManager::class.java)

  private val sampleCount: Int
    get() =
        InstrumentationRegistry.getArguments().getString("speedSamples")?.toInt()
            ?: DEFAULT_SAMPLE_COUNT

  @Before
  fun setup() {
    assertTrue("Location is switched off on the device", locationManager.isLocationEnabled)
    // Test providers are refused unless this app is the selected mock location app.
    runShell("appops set ${targetContext.packageName} android:mock_location allow")
    removeGpsTestProvider()
    locationManager.addTestProvider(
        GPS,
        ProviderProperties.Builder()
            .setAccuracy(ProviderProperties.ACCURACY_FINE)
            .setPowerUsage(ProviderProperties.POWER_USAGE_LOW)
            .build(),
    )
    locationManager.setTestProviderEnabled(GPS, true)
  }

  @After
  fun tearDown() {
    Surprise.stop()
    removeGpsTestProvider()
    MockDeviceKit.getInstance(targetContext).disable()
  }

  // Location in, speed out, with no UI involved. If this passes and the display test fails, the
  // problem is between Surprise and the screen.
  @Test
  fun surpriseSpeedTracksInjectedWalk() {
    Surprise.start(targetContext)
    walk(sampleCount) { sampleIndex, expectedSpeed ->
      awaitSpeed(sampleIndex, expectedSpeed, "Surprise") {
        Surprise.currentGroundSpeedMetersPerSecond()
      }
    }
  }

  // The full path. Mock glasses stream, which starts Surprise, and the speed chip on the camera
  // screen has to follow every step of the walk.
  @Test
  fun speedChipTracksInjectedWalk() {
    startMockStream()
    walk(sampleCount) { sampleIndex, expectedSpeed ->
      awaitSpeed(sampleIndex, expectedSpeed, "Speed chip") { displayedSpeed() }
    }
  }

  // MARK: - Walk

  // Injects an anchor fix, then sampleCount more. A fix is sent when the wall clock reaches its
  // timestamp, so no fix is ever from the future. The first fix has no predecessor and must
  // read 0.
  private fun walk(sampleCount: Int, check: (sampleIndex: Int, expectedSpeed: Double) -> Unit) {
    var latitude = START_LATITUDE
    var fixElapsedNanos = SystemClock.elapsedRealtimeNanos()
    injectFix(latitude, fixElapsedNanos)
    check(0, 0.0)

    for (sampleIndex in 1..sampleCount) {
      val speed = SPEEDS_METERS_PER_SECOND[sampleIndex % SPEEDS_METERS_PER_SECOND.size]
      val intervalMs = INTERVALS_MS[sampleIndex % INTERVALS_MS.size]

      fixElapsedNanos += intervalMs * 1_000_000L
      latitude += speed * (intervalMs / 1_000.0) / METERS_PER_DEGREE_LATITUDE

      val waitMs = (fixElapsedNanos - SystemClock.elapsedRealtimeNanos()) / 1_000_000L
      if (waitMs > 0) Thread.sleep(waitMs)
      injectFix(latitude, fixElapsedNanos)
      check(sampleIndex, speed)
    }
  }

  private fun injectFix(latitude: Double, elapsedRealtimeNanos: Long) {
    val fix =
        Location(GPS).apply {
          this.latitude = latitude
          longitude = START_LONGITUDE
          accuracy = 3f
          time = System.currentTimeMillis()
          this.elapsedRealtimeNanos = elapsedRealtimeNanos
        }
    locationManager.setTestProviderLocation(GPS, fix)
  }

  // Polls until the reading matches, then fails with the sample number and the last value seen.
  // The tolerance covers the display rounding to 2 decimals and the sphere vs ellipsoid gap.
  private fun awaitSpeed(
      sampleIndex: Int,
      expectedSpeed: Double,
      source: String,
      read: () -> Double?,
  ) {
    val tolerance = max(0.02, expectedSpeed * 0.01)
    val deadline = SystemClock.elapsedRealtime() + SPEED_TIMEOUT_MS
    var lastReading: Double? = null
    while (SystemClock.elapsedRealtime() < deadline) {
      lastReading = read()
      if (lastReading != null && abs(lastReading - expectedSpeed) <= tolerance) return
      Thread.sleep(POLL_INTERVAL_MS)
    }
    fail(
        "Sample $sampleIndex: expected %.2f m/s, $source showed ${lastReading ?: "nothing"}"
            .format(expectedSpeed)
    )
  }

  // Reads the number out of the "Speed: 1.40 m/s" chip, or null if the chip isn't on screen.
  private fun displayedSpeed(): Double? {
    val label = "${targetContext.getString(R.string.status_speed)}: "
    val chip =
        composeTestRule
            .onAllNodesWithText(label, substring = true, useUnmergedTree = true)
            .fetchSemanticsNodes()
            .firstOrNull() ?: return null
    val text = chip.config.getOrNull(SemanticsProperties.Text)?.joinToString("") { it.text }
    return text?.let { SPEED_TEXT_PATTERN.find(it)?.groupValues?.get(1)?.toDouble() }
  }

  // MARK: - Mock glasses

  // Pairs worn, powered, unfolded mock glasses with a video feed, opens the Camera tab, starts a
  // session and the preview, and waits until the stream is live.
  private fun startMockStream() {
    val mockDeviceKit = MockDeviceKit.getInstance(targetContext)
    mockDeviceKit.enable(MockDeviceKitConfig(initialPermissionsGranted = false))
    // Set explicitly because MockDeviceKit is a process singleton and a grant from another test
    // can leak in.
    mockDeviceKit.permissions.set(Permission.CAMERA, PermissionStatus.Denied)
    val glasses = mockDeviceKit.pairGlasses(GlassesModel.RAYBAN_META).getOrThrow()
    glasses.powerOn()
    glasses.don()
    glasses.unfold()
    glasses.services.camera.setCameraFeed(assetUri("plant.mp4"))

    composeTestRule.onNodeWithText("Camera").performClick()
    composeTestRule.waitUntilExactlyOneExists(
        hasTestTag("start_session_button").and(isEnabled()),
        timeoutMillis = STREAM_TIMEOUT_MS,
    )
    composeTestRule.onNodeWithTag("start_session_button").performClick()

    composeTestRule.waitUntilExactlyOneExists(
        hasTestTag("start_preview_button").and(isEnabled()),
        timeoutMillis = STREAM_TIMEOUT_MS,
    )
    composeTestRule.onNodeWithTag("start_preview_button").performClick()

    // Camera starts denied, so the first preview asks to redirect. Continue makes the mock grant it.
    val continueLabel = targetContext.getString(R.string.camera_permission_continue)
    composeTestRule.waitUntilExactlyOneExists(
        hasText(continueLabel),
        timeoutMillis = STREAM_TIMEOUT_MS,
    )
    composeTestRule.onNodeWithText(continueLabel).performClick()

    // Capture only enables at StreamState.STREAMING, which is also when Surprise starts.
    composeTestRule.waitUntilExactlyOneExists(
        hasTestTag("capture_button").and(isEnabled()),
        timeoutMillis = STREAM_TIMEOUT_MS,
    )
  }

  // MARK: - Plumbing

  private fun removeGpsTestProvider() {
    try {
      locationManager.removeTestProvider(GPS)
    } catch (notATestProvider: IllegalArgumentException) {
      // Normal on a clean run. It only succeeds when a previous run died before tearDown.
      Log.d(TAG, "No GPS test provider to remove: ${notATestProvider.message}")
    }
  }

  // Reads the output to the end so the command has finished before the next line runs.
  private fun runShell(command: String) {
    val output = InstrumentationRegistry.getInstrumentation().uiAutomation.executeShellCommand(command)
    ParcelFileDescriptor.AutoCloseInputStream(output).use { it.readBytes() }
  }

  private fun assetUri(assetName: String): Uri {
    val outFile = File(targetContext.cacheDir, assetName)
    InstrumentationRegistry.getInstrumentation().context.assets.open(assetName).use { input ->
      FileOutputStream(outFile).use { output -> input.copyTo(output) }
    }
    return Uri.fromFile(outFile)
  }
}
