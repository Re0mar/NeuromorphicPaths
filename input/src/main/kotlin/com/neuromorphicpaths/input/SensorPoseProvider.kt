package com.neuromorphicpaths.input

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import com.neuromorphicpaths.core.CameraPose
import com.neuromorphicpaths.core.PoseProvider

/**
 * Camera pitch and roll from the phone's fused rotation vector.
 *
 * Assumes the phone is held upright with the back camera facing forward, the camera app posture.
 * Yaw relative to the walker cannot come from a sensor, so it is fixed at construction, and so
 * is the height.
 */
class SensorPoseProvider(
    context: Context,
    private val heightMeters: Double,
    private val yawRadians: Double = 0.0,
) : PoseProvider, SensorEventListener, AutoCloseable {

    private val sensorManager: SensorManager = context.getSystemService(SensorManager::class.java)
    private val rotationVector: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val rotation = FloatArray(9)
    private val remapped = FloatArray(9)
    private val orientation = FloatArray(3)

    @Volatile
    private var latest = CameraPose(pitchRadians = 0.0, yawRadians = yawRadians, rollRadians = 0.0, heightMeters = heightMeters)

    @Volatile
    private var latestAzimuthRadians: Double? = null

    /** True when the phone has the sensor. Without it the pose stays level. */
    val available: Boolean get() = rotationVector != null

    fun start() {
        rotationVector?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    override fun currentPose(): CameraPose = latest

    /**
     * Where the camera points over the ground, in radians, from the same rotation vector. Null
     * until the sensor has reported. The reference direction and the sign are the platform's, and
     * only the changes over time are used, so neither is documented further.
     */
    fun currentAzimuthRadians(): Double? = latestAzimuthRadians

    override fun onSensorChanged(event: SensorEvent) {
        SensorManager.getRotationMatrixFromVector(rotation, event.values)
        // Upright phone: the screen's outward axis becomes the new Y. This is the remap the
        // platform documents for camera apps.
        SensorManager.remapCoordinateSystem(rotation, SensorManager.AXIS_X, SensorManager.AXIS_Z, remapped)
        SensorManager.getOrientation(remapped, orientation)
        // After the remap, tipping the camera toward the ground raises the screen's outward axis,
        // which the platform reports as negative pitch. CameraPose wants looking down positive.
        latestAzimuthRadians = orientation[0].toDouble()
        latest = CameraPose(
            pitchRadians = -orientation[1].toDouble(),
            yawRadians = yawRadians,
            rollRadians = orientation[2].toDouble(),
            heightMeters = heightMeters,
        )
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    override fun close() {
        sensorManager.unregisterListener(this)
    }
}
