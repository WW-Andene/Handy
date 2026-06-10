package com.arhand.ui

import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.arhand.util.PerfState
import com.arhand.tracking.Gesture

/**
 * Top-right HUD overlay — FPS / inference ms / skip ratio.
 *
 * E4 — Debug mode toggle:
 * Long-press the HUD area to cycle between MINIMAL (hidden) and VERBOSE mode.
 * VERBOSE adds landmark count, OEF alpha, slot assignments.
 * Gated by BuildConfig.DEBUG at runtime.
 */
enum class HudMode { MINIMAL, VERBOSE }

@Composable
fun HudOverlay(
    perf: PerfState,
    renderMode: String,
    torchOn: Boolean,
    landmarkCount: Int = 0,
    activeSlots: String = "--",
    activeGesture: Gesture? = null
) {
    // E4: Long-press cycles MINIMAL → VERBOSE → MINIMAL
    var hudMode by remember { mutableStateOf(HudMode.VERBOSE) }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                detectTapGestures(onLongPress = {
                    hudMode = if (hudMode == HudMode.VERBOSE) HudMode.MINIMAL else HudMode.VERBOSE
                })
            }
    ) {
        if (hudMode == HudMode.VERBOSE) {
            Column(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 48.dp, end = 16.dp),
                horizontalAlignment = Alignment.End
            ) {
                HudLabel("FPS",  "%.0f".format(perf.fps),  if (perf.fps < 30f) Warn else Plasma)
                HudLabel("INF",  "%.1fms".format(perf.inferenceMs), Plasma)
                HudLabel("SKIP", "%.0f%%".format(perf.skipRatio * 100f), Plasma)
                Spacer(Modifier.height(8.dp))
                HudLabel("MODE",  renderMode, Plasma)
                HudLabel("TORCH", if (torchOn) "ON" else "OFF", if (torchOn) Plasma else Warn)
                // E4: Verbose-only stats
                Spacer(Modifier.height(8.dp))
                HudLabel("LMS",   landmarkCount.toString(), Plasma.copy(alpha = 0.7f))
                HudLabel("SLOTS", activeSlots,              Plasma.copy(alpha = 0.7f))
                // G4: Active gesture display
                val gestureColor = if (activeGesture != null) Color(0xFF00E676) else Plasma.copy(alpha = 0.4f)
                HudLabel(
                    "GESTURE",
                    activeGesture?.let { "${it.symbol} ${it.label}" } ?: "--",
                    gestureColor
                )
            }
        }
    }
}

@Composable
fun HudLabel(key: String, value: String, color: Color) {
    Row(horizontalArrangement = Arrangement.End) {
        Text(
            text = "$key ",
            fontSize = 10.sp,
            color = Color.White.copy(alpha = 0.4f),
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Normal
        )
        Text(
            text = value,
            fontSize = 10.sp,
            color = color,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold
        )
    }
}

