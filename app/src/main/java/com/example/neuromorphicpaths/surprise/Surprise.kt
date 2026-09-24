package com.example.neuromorphicpaths.surprise

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import androidx.core.content.ContextCompat
import java.util.concurrent.atomic.AtomicReference

/** Tracks ground speed from successive GPS coordinates. */
object Surprise {
  private val speedMetersPerSecond = AtomicReference(0.0)
  private var locationManager: LocationManager? = null
  private var previousLocation: Location? = null

  private val locationListener =
      LocationListener { location ->
        val previous = previousLocation
        if (previous != null) {
          val elapsedSeconds = (location.elapsedRealtimeNanos - previous.elapsedRealtimeNanos) / 1e9
          if (elapsedSeconds > 0.0) {
            speedMetersPerSecond.set(previous.distanceTo(location) / elapsedSeconds)
          }
        }
        previousLocation = location
      }

  fun start(context: Context) {
    stop()
    if (
        ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) !=
            PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_COARSE_LOCATION) !=
                PackageManager.PERMISSION_GRANTED
    ) {
      return
    }

    val manager = context.getSystemService(LocationManager::class.java)
    locationManager = manager
    previousLocation = null
    speedMetersPerSecond.set(0.0)
    manager.requestLocationUpdates(
        LocationManager.GPS_PROVIDER,
        1_000L,
        0f,
        locationListener,
        Looper.getMainLooper(),
    )
  }

  fun stop() {
    locationManager?.removeUpdates(locationListener)
    locationManager = null
    previousLocation = null
    speedMetersPerSecond.set(0.0)
  }

  /** Returns the latest GPS ground speed in meters per second. */
  fun currentGroundSpeedMetersPerSecond(): Double = speedMetersPerSecond.get()
}
