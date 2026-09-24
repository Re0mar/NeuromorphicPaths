package com.example.sidewalkvision

import android.graphics.BitmapFactory
import android.graphics.Bitmap
import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*

@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.example.sidewalkvision", appContext.packageName)
    }

    @Test
    fun testPathDetectionOnAssetImage() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val inputStream = appContext.assets.open("testimage.png")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val argbBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        
        val detector = PathDetector(appContext)
        val result = detector.detect(argbBitmap)
        
        Log.d("PathDetectorTest", "Detection score: ${result.score}, box: ${result.box}, hasMask: ${result.maskBitmap != null}")
        
        assertNotNull(result)
        assertTrue("Path should be detected with score > 0.015, got ${result.score}", result.score >= 0.015f)
        assertNotNull("Mask bitmap should not be null", result.maskBitmap)
        
        detector.close()
    }
}
