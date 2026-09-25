package com.neuromorphicpaths.input

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * The walker's ground speed from GPS, smoothed. Null until the first usable fix.
 *
 * Feeds the walker state the pipeline hands to the guidance field, which otherwise assumes a
 * fixed walking pace. GPS alone is used, since the fused provider needs Play services and the
 * phone is outdoors when speed matters. Indoors this stays null and the field keeps its default.
 */
class GpsGroundSpeed(private val context: Context) : AutoCloseable {
    private val speedState = MutableStateFlow<Double?>(null)
    val metersPerSecond: StateFlow<Double?> = speedState.asStateFlow()

    private val smoother = SpeedSmoother()
    private var locationManager: LocationManager? = null
    private var previousLocation: Location? = null
    private val locationListener = LocationListener { location -> onLocation(location) }

    fun hasPermission(): Boolean =
        context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    /** Starts listening. Returns false, and does nothing, without location permission. */
    fun start(): Boolean {
        stop()
        if (!hasPermission()) return false
        val manager = context.getSystemService(LocationManager::class.java) ?: return false
        locationManager = manager
        try {
            manager.requestLocationUpdates(
                LocationManager.GPS_PROVIDER,
                UPDATE_INTERVAL_MILLIS,
                MINIMUM_DISTANCE_METERS,
                locationListener,
                Looper.getMainLooper(),
            )
        } catch (revoked: SecurityException) {
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
        speedState.value = null
    }

    override fun close() = stop()

    private fun onLocation(location: Location) {
        val previous = previousLocation
        val reading = rawSpeedMetersPerSecond(
            dopplerMetersPerSecond = if (location.hasSpeed()) location.speed.toDouble() else null,
            distanceMeters = previous?.distanceTo(location)?.toDouble(),
            elapsedSeconds = previous?.let { (location.elapsedRealtimeNanos - it.elapsedRealtimeNanos) / NANOS_PER_SECOND },
        )
        previousLocation = location
        if (reading != null) {
            speedState.value = smoother.add(reading, location.elapsedRealtimeNanos)
        }
    }

    private companion object {
        const val UPDATE_INTERVAL_MILLIS = 1_000L
        const val MINIMUM_DISTANCE_METERS = 0f
        const val NANOS_PER_SECOND = 1e9
    }
}
