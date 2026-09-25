package com.neuromorphicpaths.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.neuromorphicpaths.app.DetectorChoice
import com.neuromorphicpaths.output.GuidanceOverlay
import com.neuromorphicpaths.output.ScreenOverlayDisplay

/** Two rows of controls over the overlay. Everything the first version lets a person do. */
@Composable
fun MainScreen(
    display: ScreenOverlayDisplay,
    detectorChoice: DetectorChoice,
    yoloWorldAvailable: Boolean,
    onDetectorChoiceChange: (DetectorChoice) -> Unit,
    onStartCamera: () -> Unit,
    onOpenVideo: () -> Unit,
) {
    Column(Modifier.fillMaxSize()) {
        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 4.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Button(onClick = onStartCamera) { Text("Camera") }
            Button(onClick = onOpenVideo) { Text("Open video") }
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
        GuidanceOverlay(
            display = display,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
        )
    }
}
