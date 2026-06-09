package com.arhand.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Phase 2 placeholder — ARCore Depth API panel.
 * Shows real-time depth confidence and toggle for depth mesh overlay.
 * Active only when ARCore depth is available.
 */
@Composable
fun DepthScannerPanel(
    depthMode: Boolean,
    depthConfidence: Float,
    onToggleDepth: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(UIBg)
            .padding(horizontal = 16.dp, vertical = 10.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column {
                Text("DEPTH API", fontSize = 9.sp, color = Color.White.copy(0.4f),
                    fontFamily = FontFamily.Monospace)
                Text(
                    text = if (depthMode) "ACTIVE" else "PHASE 2",
                    fontSize = 12.sp,
                    color = if (depthMode) Plasma else Color.White.copy(0.3f),
                    fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold
                )
            }

            if (depthMode) {
                Column(horizontalAlignment = Alignment.End) {
                    Text("CONFIDENCE", fontSize = 9.sp, color = Color.White.copy(0.4f),
                        fontFamily = FontFamily.Monospace)
                    Text("%.0f%%".format(depthConfidence * 100f), fontSize = 12.sp, color = Plasma,
                        fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                }
            }

            CtrlButton(
                label = if (depthMode) "DEPTH\nON" else "DEPTH\nOFF",
                active = depthMode,
                onClick = onToggleDepth
            )
        }
    }
}
