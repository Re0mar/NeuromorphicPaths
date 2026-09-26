package com.neuromorphicpaths.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.Button
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.neuromorphicpaths.app.DetectorChoice
import com.neuromorphicpaths.output.FloatingForm
import com.neuromorphicpaths.output.GuidanceOverlay
import com.neuromorphicpaths.output.ScreenOverlayDisplay

/** Three rows of controls over the overlay: what to run, which detector, and what floats over other apps. */
@Composable
fun MainScreen(
    display: ScreenOverlayDisplay,
    detectorChoice: DetectorChoice,
    yoloWorldAvailable: Boolean,
    segmenterOn: Boolean,
    segmenterAvailable: Boolean,
    running: Boolean,
    floatingForm: FloatingForm,
    onDetectorChoiceChange: (DetectorChoice) -> Unit,
    onSegmenterChange: (Boolean) -> Unit,
    onStartCamera: () -> Unit,
    onOpenRecording: () -> Unit,
    onStop: () -> Unit,
    onFloatingFormChange: (FloatingForm) -> Unit,
) {
    // Edge-to-edge is the default now, so without this the buttons sit under the status bar and
    // a tap on them lands on the clock instead.
    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(onClick = onStartCamera) { Text("Camera") }
            Button(onClick = onOpenRecording) { Text("Open recording") }
            Button(onClick = onStop, enabled = running) { Text("Stop") }
            // The segmenter is a second model beside the detector, so it gets a switch rather than
            // a place in the detector row. Grayed out on a build without its model file.
            Text("Surfaces")
            Switch(checked = segmenterOn, onCheckedChange = onSegmenterChange, enabled = segmenterAvailable)
        }
        SingleChoiceSegmentedButtonRow(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 4.dp),
        ) {
            val choices = DetectorChoice.entries
            choices.forEachIndexed { index, choice ->
                SegmentedButton(
                    selected = choice == detectorChoice,
                    onClick = { onDetectorChoiceChange(choice) },
                    shape = SegmentedButtonDefaults.itemShape(index = index, count = choices.size),
                    enabled = choice != DetectorChoice.YOLO_WORLD || yoloWorldAvailable,
                ) {
                    Text(choice.label)
                }
            }
        }
        // What floats over other apps once this screen is left. Off, a symbol in the corner, or
        // the path's ribbon across the whole screen.
        SingleChoiceSegmentedButtonRow(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 4.dp),
        ) {
            val forms = FloatingForm.entries
            forms.forEachIndexed { index, form ->
                SegmentedButton(
                    selected = form == floatingForm,
                    onClick = { onFloatingFormChange(form) },
                    shape = SegmentedButtonDefaults.itemShape(index = index, count = forms.size),
                ) {
                    Text(FLOATING_FORM_LABELS.getValue(form))
                }
            }
        }
        GuidanceOverlay(
            display = display,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
        )
    }
}

private val FLOATING_FORM_LABELS = mapOf(
    FloatingForm.OFF to "No overlay",
    FloatingForm.CORNER to "Corner",
    FloatingForm.FULL_SCREEN to "Full screen",
)
