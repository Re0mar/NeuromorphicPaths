package com.neuromorphicpaths.input

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.sqrt

/**
 * The walker's speed from their steps, live, off the phone's accelerometer.
 *
 * Needs no permission and works indoors, which GPS does not, so this is the speed the field
 * uses and GPS is the outdoor cross-check. The accelerometer reports gravity included, and the
 * estimator takes the magnitude with gravity removed, the same quantity the recording's log
 * gives on replay, so live and replay run the same code.
 */
class AccelerometerCadenceSpeed(
    context: Context,
    private val estimator: StepCadenceSpeedEstimator = StepCadenceSpeedEstimator(),
) : SensorEventListener, AutoCloseable {

    private val sensorManager: SensorManager = context.getSystemService(SensorManager::class.java)
    private val accelerometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val speedState = MutableStateFlow<Double?>(null)
    val metersPerSecond: StateFlow<Double?> = speedState.asStateFlow()

    /** True when the phone has the sensor. Without it the speed stays null and the field keeps its default. */
    val available: Boolean get() = accelerometer != null

    fun start() {
        accelerometer?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val magnitude = sqrt(event.values[0] * event.values[0] + event.values[1] * event.values[1] + event.values[2] * event.values[2]).toDouble()
        speedState.value = estimator.add(event.timestamp, magnitude - AccelerationLog.GRAVITY_METERS_PER_SECOND_SQUARED)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    override fun close() {
        sensorManager.unregisterListener(this)
        estimator.reset()
        speedState.value = null
    }
}
