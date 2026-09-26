package com.neuromorphicpaths.output

import android.content.Context
import android.graphics.PixelFormat
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.unit.dp
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import com.neuromorphicpaths.core.GuidanceDisplay
import com.neuromorphicpaths.core.GuidanceUpdate
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch

/** Which floating form is up over other apps. */
enum class FloatingForm {
    OFF,
    CORNER,
    FULL_SCREEN,
}

/**
 * The guidance drawn over other apps, in a window of its own, so the phone can be used for
 * something else while the pipeline runs.
 *
 * Two forms. The corner form is one thick symbol in the top right corner, the way a navigation
 * app shows the next turn: straight, bear, hard turn, STOP or U-turn, blue to red with surprise.
 * Tapping it opens the full-screen form. The full-screen form is the projected path's ribbon
 * alone, drawn across the whole screen at [GuidanceDisplayTuning.fullScreenNormalOpacity], raised
 * to [GuidanceDisplayTuning.fullScreenEmergencyOpacity] when surprise is red or STOP is up. The
 * ribbon is drawn at full strength inside that window, fading only with distance, so those two
 * numbers are exactly how strong it looks at the walker's feet. It lets every touch
 * through to the app underneath, so its way back to the corner form is a button in a small
 * window of its own in the lower right.
 *
 * Nothing is drawn while the app's own screen is showing, since that screen draws the same
 * thing, and nothing is drawn without the draw-over-apps permission. The windows need a
 * lifecycle and a saved-state owner to host Compose outside an activity, which the service
 * running the pipeline provides.
 */
class FloatingGuidanceDisplay(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val savedStateRegistryOwner: SavedStateRegistryOwner,
    scope: CoroutineScope,
    private val tuning: GuidanceDisplayTuning = GuidanceDisplayTuning(),
    mainDispatcher: CoroutineDispatcher = Dispatchers.Main,
) : GuidanceDisplay {

    override val name: String = "floating"

    private val windowManager = context.getSystemService(WindowManager::class.java)
    private val advisor = TurnAdvisor(tuning)
    private val latestUpdate = MutableStateFlow<GuidanceUpdate?>(null)
    private val latestAdvice = MutableStateFlow(TurnAdvice.CALM)
    private val formState = MutableStateFlow(FloatingForm.OFF)
    private val appVisible = MutableStateFlow(false)
    private val openWindows = mutableListOf<View>()
    private var ribbonWindow: View? = null
    private var ribbonParams: WindowManager.LayoutParams? = null

    /** The form the person picked. What is on screen is this, unless the app itself is showing. */
    val form: StateFlow<FloatingForm> = formState.asStateFlow()

    init {
        scope.launch(mainDispatcher) {
            combine(formState, appVisible) { picked, visible -> if (visible) FloatingForm.OFF else picked }
                .distinctUntilChanged()
                .collect(::openForm)
        }
        scope.launch(mainDispatcher) {
            latestAdvice.map { it.emergency }.distinctUntilChanged().collect(::setRibbonOpacity)
        }
    }

    fun setForm(form: FloatingForm) {
        formState.value = form
    }

    /** Called by whoever knows whether the app's own screen is in front. */
    fun setAppVisible(visible: Boolean) {
        appVisible.value = visible
    }

    override suspend fun show(update: GuidanceUpdate) {
        latestAdvice.value = advisor.advise(update.guidance, update.walker.speedMetersPerSecond, update.frame.timestampNanos)
        latestUpdate.value = update
    }

    /** Takes the windows down. Call from the main thread, as the service does in its own teardown. */
    override fun close() {
        closeWindows()
        latestUpdate.value = null
    }

    private fun openForm(form: FloatingForm) {
        closeWindows()
        if (form == FloatingForm.OFF) return
        if (!Settings.canDrawOverlays(context)) {
            // The app asks for the permission when the form is picked, so this is someone who
            // revoked it in settings afterwards.
            Log.w(TAG, "No draw-over-apps permission, floating form $form not shown")
            return
        }
        when (form) {
            FloatingForm.CORNER -> addWindow(composeView { CornerSymbol() }, cornerParams())
            FloatingForm.FULL_SCREEN -> {
                val params = fullScreenParams()
                val view = composeView { Ribbon() }
                addWindow(view, params)
                ribbonWindow = view
                ribbonParams = params
                addWindow(composeView { SmallFormButton() }, buttonParams())
            }
            FloatingForm.OFF -> Unit
        }
    }

    private fun closeWindows() {
        for (view in openWindows) windowManager.removeView(view)
        openWindows.clear()
        ribbonWindow = null
        ribbonParams = null
    }

    private fun addWindow(view: View, params: WindowManager.LayoutParams) {
        windowManager.addView(view, params)
        openWindows += view
    }

    private fun setRibbonOpacity(emergency: Boolean) {
        val view = ribbonWindow ?: return
        val params = ribbonParams ?: return
        params.alpha = if (emergency) tuning.fullScreenEmergencyOpacity else tuning.fullScreenNormalOpacity
        windowManager.updateViewLayout(view, params)
    }

    private fun composeView(content: @Composable () -> Unit): ComposeView = ComposeView(context).apply {
        setViewTreeLifecycleOwner(lifecycleOwner)
        setViewTreeSavedStateRegistryOwner(savedStateRegistryOwner)
        setContent(content)
    }

    @Composable
    private fun CornerSymbol() {
        val advice by latestAdvice.collectAsState()
        Box(
            Modifier
                .size(tuning.cornerSymbolSizeDp.dp)
                .background(Color.Black.copy(alpha = CORNER_BACKGROUND_ALPHA), RoundedCornerShape(16.dp))
                .clickable { setForm(FloatingForm.FULL_SCREEN) }
                .padding(10.dp),
        ) {
            Canvas(Modifier.fillMaxSize()) { drawTurnSymbol(advice.symbol, pathColor(advice.alertFraction)) }
        }
    }

    @Composable
    private fun Ribbon() {
        val update by latestUpdate.collectAsState()
        val advice by latestAdvice.collectAsState()
        val current = update ?: return
        Canvas(Modifier.fillMaxSize()) {
            // Laid out as if the camera frame filled the screen, so the ribbon sits where it
            // would on the app's own screen.
            val fit = FittedImage(current.frame.width, current.frame.height, size)
            // Full strength at the feet, so the window's own opacity is what the eye gets there.
            drawProjectedPath(current, fit, pathColor(advice.alertFraction), fadeByInformation = false)
        }
    }

    @Composable
    private fun SmallFormButton() {
        Button(onClick = { setForm(FloatingForm.CORNER) }) { Text("Small") }
    }

    // The full-screen window ignores touches, so the app underneath keeps working. Its opacity
    // has to stay at or below Android's touch-through limit, which the tuning checks.
    private fun fullScreenParams() = WindowManager.LayoutParams(
        WindowManager.LayoutParams.MATCH_PARENT,
        WindowManager.LayoutParams.MATCH_PARENT,
        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
        PixelFormat.TRANSLUCENT,
    ).apply {
        alpha = if (latestAdvice.value.emergency) tuning.fullScreenEmergencyOpacity else tuning.fullScreenNormalOpacity
    }

    // The small windows take touches only inside their own bounds, and never the keyboard.
    private fun smallWindowParams(gravity: Int, marginXDp: Int, marginYDp: Int) = WindowManager.LayoutParams(
        WindowManager.LayoutParams.WRAP_CONTENT,
        WindowManager.LayoutParams.WRAP_CONTENT,
        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
        PixelFormat.TRANSLUCENT,
    ).apply {
        this.gravity = gravity
        val density = context.resources.displayMetrics.density
        x = (marginXDp * density).toInt()
        y = (marginYDp * density).toInt()
    }

    // Below the status bar, where a navigation app puts its next-turn card.
    private fun cornerParams() = smallWindowParams(Gravity.TOP or Gravity.END, marginXDp = 12, marginYDp = 64)

    // Above the navigation bar.
    private fun buttonParams() = smallWindowParams(Gravity.BOTTOM or Gravity.END, marginXDp = 16, marginYDp = 72)

    private companion object {
        const val TAG = "FloatingGuidance"
        const val CORNER_BACKGROUND_ALPHA = 0.6f
    }
}
