package com.example.sidewalkvision

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.exp

private const val UPDATE_INTERVAL_MS = 1_000L

// The same one-second window docs/math/surprise_and_tau.md uses for the rider's running average.
private const val SMOOTHING_WINDOW_SECONDS = 1.0

/**
 * One speed reading from a GPS fix, in meters per second, or null when there is nothing to go on.
 *
 * Prefers the receiver's own speed, which Android derives from the Doppler shift of the satellite
 * signals. Distance between two fixes over the time between them is the fallback, because phone
 * positions wobble by a few meters from fix to fix, more than a walker covers in one second.
 */
fun rawSpeedMetersPerSecond(
    dopplerMetersPerSecond: Float?,
    distanceMeters: Float?,
    elapsedSeconds: Double?,
): Double? {
    if (dopplerMetersPerSecond != null) return dopplerMetersPerSecond.toDouble()
    if (distanceMeters == null || elapsedSeconds == null || elapsedSeconds <= 0.0) return null
    return distanceMeters / elapsedSeconds
}

/**
 * Running average of speed readings with weight 1 - exp(-dt / window), so the result is the same
 * whether fixes arrive every second or every few.
 */
class SpeedSmoother(private val windowSeconds: Double = SMOOTHING_WINDOW_SECONDS) {
    var metersPerSecond: Double? = null
        private set
    private var previousElapsedNanos: Long? = null

    fun add(readingMetersPerSecond: Double, elapsedNanos: Long): Double {
        val current = metersPerSecond
        val previousNanos = previousElapsedNanos
        if (current == null || previousNanos == null) {
            metersPerSecond = readingMetersPerSecond
            previousElapsedNanos = elapsedNanos
            return readingMetersPerSecond
        }
        val elapsedSeconds = (elapsedNanos - previousNanos) / 1e9
        // A repeated or out-of-order fix carries no new time, so it can't move the average.
        if (elapsedSeconds <= 0.0) return current

        val weight = 1.0 - exp(-elapsedSeconds / windowSeconds)
        val smoothed = current + weight * (readingMetersPerSecond - current)
        metersPerSecond = smoothed
        previousElapsedNanos = elapsedNanos
        return smoothed
    }

    fun reset() {
        metersPerSecond = null
        previousElapsedNanos = null
    }
}

/** Ground speed from GPS, smoothed. Null until the first usable fix. */
class GroundSpeedTracker(private val context: Context) {
    private val _metersPerSecond = MutableStateFlow<Double?>(null)
    val metersPerSecond: StateFlow<Double?> = _metersPerSecond.asStateFlow()

    private val smoother = SpeedSmoother()
    private var locationManager: LocationManager? = null
    private var previousLocation: Location? = null
    private val locationListener = LocationListener { location -> onLocation(location) }

    fun hasPermission(): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED ||
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED

    /** Starts listening. Returns false, and does nothing, without location permission. */
    fun start(): Boolean {
        stop()
        if (!hasPermission()) return false
        val manager = context.getSystemService(LocationManager::class.java) ?: return false
        locationManager = manager
        try {
            manager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                UPDATE_INTERVAL_MS,
                0f,
                locationListener,
                Looper.getMainLooper(),
            )
        } catch (securityException: SecurityException) {
            // Permission was revoked between the check and the request. Report it as not started.
            locationManager = null
            return false
        }
        return true
    }

    fun stop() {
        locationManager?.removeUpdates(locationListener)
        locationManager = null
        previousLocation = null
        smoother.reset()
        _metersPerSecond.value = null
    }

    private fun onLocation(location: Location) {
        val previous = previousLocation
        val reading = rawSpeedMetersPerSecond(
            dopplerMetersPerSecond = if (location.hasSpeed()) location.speed else null,
            distanceMeters = previous?.distanceTo(location),
            elapsedSeconds = previous?.let { (location.elapsedRealtimeNanos - it.elapsedRealtimeNanos) / 1e9 },
        )
        previousLocation = location
        if (reading != null) {
            _metersPerSecond.value = smoother.add(reading, location.elapsedRealtimeNanos)
        }
    }
}
