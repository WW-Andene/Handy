package com.arhand.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.arhand.util.PerfState

val Plasma = Color(0xFF00FFE5)
val Warn   = Color(0xFFFF4060)

/**
 * Top-right HUD overlay — FPS / inference ms / skip ratio.
 * Port of the HTML prototype's stats overlay.
 */
@Composable
fun HudOverlay(perf: PerfState, renderMode: String, torchOn: Boolean) {
    Box(modifier = Modifier.fillMaxSize()) {
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
