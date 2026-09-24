package com.example.sidewalkvision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

// The focal length analysis/ assumes for the Belgian dataset, 1450 px at 1920 wide. testimage.png
// is one of its frames, so the app's pose here can be compared with the Python one directly.
private val DATASET_INTRINSICS = CameraIntrinsics(1450.0 / 1920)

@RunWith(AndroidJUnit4::class)
class PoseOnDeviceTest {
    @Test
    fun theTestImageGetsAPoseWithEveryAngle() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val image = context.assets.open("testimage.png").use { BitmapFactory.decodeStream(it) }
            .copy(Bitmap.Config.ARGB_8888, true)
        val detector = PathDetector(context)

        val pose = detector.detect(image, DATASET_INTRINSICS).pose
        detector.close()

        assertNotNull(pose)
        Log.i(
            "PoseOnDeviceTest",
            "status=${pose!!.status} reliable=${pose.reliable} reason=${pose.unreliableReason} " +
                "pitch=${pose.pitchDegrees} heading=${pose.headingDegrees} position=${pose.positionAcross} " +
                "height=${pose.cameraHeightMeters} leftRmsCells=${pose.left?.rmsCells} rightRmsCells=${pose.right?.rmsCells} " +
                "leftPoints=${pose.left?.pointsUsed}/${pose.left?.pointsOffered} rightPoints=${pose.right?.pointsUsed}/${pose.right?.pointsOffered}"
        )
        assertEquals(PoseStatus.OK, pose.status)
        assertNotNull(pose.pitchDegrees)
        assertNotNull(pose.headingDegrees)
        assertNotNull(pose.cameraHeightMeters)
    }
}
