package com.example.sidewalkvision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

// testimage.png is a sidewalk seen from low down, so near the bottom it runs off both sides.
private const val TEST_IMAGE = "testimage.png"

// A clipped point sits within one mask cell of the side. At 1920 wide a cell is 24 px, 1.25%.
private const val SIDE_FRACTION = 0.02f

@RunWith(AndroidJUnit4::class)
class EdgeClippingAndFrameDumpTest {
    private lateinit var context: Context
    private lateinit var dumpFolder: File

    @Before
    fun clearDumpFolder() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        dumpFolder = context.getExternalFilesDir(PathDetector.DEBUG_DUMP_FOLDER)!!
        dumpFolder.listFiles()?.forEach { it.delete() }
    }

    @After
    fun removeDumpedFrames() {
        dumpFolder.listFiles()?.forEach { it.delete() }
    }

    private fun loadTestImage(): Bitmap =
        context.assets.open(TEST_IMAGE).use { BitmapFactory.decodeStream(it) }
            .copy(Bitmap.Config.ARGB_8888, true)

    @Test
    fun aSidewalkFillingTheFramesWidthStillGetsAMask() {
        // testimage.png's box is 1.0011 wide, just over the frame. That once read as pixel
        // coordinates and emptied the mask, while the result still reported a mask bitmap.
        val detector = PathDetector(context)
        val result = detector.detect(loadTestImage())
        detector.close()

        val box = result.box!!
        assertTrue("box should span most of the frame, got ${box.toList()}", box[2] - box[0] > 0.5f)
        val mask = result.maskBitmap!!
        val pixels = IntArray(mask.width * mask.height)
        mask.getPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)
        val visible = pixels.count { (it ushr 24) != 0 }
        assertTrue("mask should cover a good part of the frame, got $visible pixels", visible > pixels.size / 10)
    }

    @Test
    fun edgePointsAtTheImageSideAreMarkedClipped() {
        val detector = PathDetector(context)
        val result = detector.detect(loadTestImage())
        detector.close()

        val clipped = result.edgePoints.filter { it.clipped }
        val unclipped = result.edgePoints.filterNot { it.clipped }
        val xs = result.edgePoints.map { it.position.x }
        assertTrue(
            "expected the sidewalk to run off the image near the bottom, got ${result.edgePoints.size} " +
                "points with x from ${xs.minOrNull()} to ${xs.maxOrNull()}",
            clipped.isNotEmpty()
        )
        assertTrue("expected edge points inside the image too", unclipped.isNotEmpty())
        for (point in clipped) {
            val nearSide = point.position.x <= SIDE_FRACTION || point.position.x >= 1f - SIDE_FRACTION
            assertTrue("clipped point at x=${point.position.x} is not at a side", nearSide)
        }
        for (point in unclipped) {
            val atSide = point.position.x <= 0f || point.position.x >= 1f - 1f / 1920f
            assertTrue("unclipped point at x=${point.position.x} sits on the border", !atSide)
        }
    }

    @Test
    fun everyThirtiethDetectionSavesTheRawAndModelInputFrames() {
        val detector = PathDetector(context)
        val image = loadTestImage()
        // Detections 0 and 30 are saved. 31 calls cover both and nothing more.
        repeat(31) { detector.detect(image) }
        detector.close()

        val saved = dumpFolder.listFiles()!!.map { it.name }.sorted()
        assertEquals(4, saved.size)
        assertEquals(2, saved.count { it.endsWith("_raw.jpg") })
        assertEquals(2, saved.count { it.endsWith("_input.jpg") })
        assertTrue(saved.all { it.startsWith("session_") && "_frame_00" in it })
    }
}
