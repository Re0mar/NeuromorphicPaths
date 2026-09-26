package com.neuromorphicpaths.input

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class AccelerationLogTest {

    // Sensor Logger's column order, phone lying flat then bumped: gravity on z, then a 2 m/s squared kick.
    private val csv = """
        time,seconds_elapsed,z,y,x
        2000000000,1.0,9.81,0,0
        1000000000,0.0,9.81,0,0
        3000000000,2.0,11.81,0,0
    """.trimIndent()

    @Test
    fun parsesByColumnNameTakesGravityOffAndSortsByTime() {
        val log = AccelerationLog.parse(csv)

        assertEquals(listOf(1_000_000_000L, 2_000_000_000L, 3_000_000_000L), log.samples.map { it.epochNanos })
        assertEquals(0.0, log.samples[0].magnitudeMinusGravity, 1e-9)
        assertEquals(2.0, log.samples[2].magnitudeMinusGravity, 1e-9)
    }

    @Test
    fun magnitudeDoesNotCareWhichAxisCarriesGravity() {
        val log = AccelerationLog.parse("time,seconds_elapsed,z,y,x\n1,0,0,9.81,0\n2,0,0,0,-9.81\n")

        assertEquals(0.0, log.samples[0].magnitudeMinusGravity, 1e-9)
        assertEquals(0.0, log.samples[1].magnitudeMinusGravity, 1e-9)
    }

    @Test
    fun aMissingColumnIsAnError() {
        assertThrows(IllegalArgumentException::class.java) { AccelerationLog.parse("time,seconds_elapsed,z,y\n1,0,9.81,0\n") }
    }
}
