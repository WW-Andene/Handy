package com.arhand.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Scan result modal — shown when scan completes with GLB export path and stats.
 * Port of the "Export Modal" from the HTML prototype.
 *
 * Includes a "VIEW 3D" button that pushes a full-screen [ModelViewerScreen]
 * over the modal when [meshPositions] is non-empty.
 */
@Composable
fun ScanResultModal(
    pointCount: Int,
    glbPath: String?,
    poseScores: List<Float>,
    meshPositions: FloatArray,
    onShare: () -> Unit,
    onRescan: () -> Unit,
    onDismiss: () -> Unit
) {
    var showViewer by remember { mutableStateOf(false) }

    if (showViewer) {
        ModelViewerScreen(
            meshPositions = meshPositions,
            onDismiss     = { showViewer = false }
        )
        return
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.75f)),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth(0.88f)
                .background(Color(0xFF0D1117), RoundedCornerShape(16.dp))
                .border(1.dp, Plasma.copy(alpha = 0.3f), RoundedCornerShape(16.dp))
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("SCAN COMPLETE", fontSize = 16.sp, color = Plasma,
                fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(16.dp))

            // Stats grid
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatCell("POINTS", pointCount.toString())
                StatCell("POSES",  "8 / 8")
                StatCell("QUALITY", "%.0f%%".format(poseScores.average() * 100f))
            }

            Spacer(Modifier.height(16.dp))

            // Pose quality bar strip
            Text("POSE QUALITY", fontSize = 9.sp, color = Color.White.copy(0.4f),
                fontFamily = FontFamily.Monospace)
            Spacer(Modifier.height(6.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                poseScores.forEach { score ->
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(24.dp)
                            .clip(RoundedCornerShape(3.dp))
                            .background(Plasma.copy(alpha = score.coerceIn(0f, 1f)))
                    )
                }
            }

            Spacer(Modifier.height(20.dp))

            // Export path
            if (glbPath != null) {
                Text(
                    text = "GLB saved to Documents",
                    fontSize = 10.sp, color = Color.White.copy(0.5f),
                    fontFamily = FontFamily.Monospace
                )
                Spacer(Modifier.height(12.dp))
            }

            // Primary actions
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                ResultButton("SHARE GLB", Modifier.weight(1f), Plasma, onShare)
                ResultButton("RE-SCAN",   Modifier.weight(1f), Color.White.copy(0.6f), onRescan)
            }

            // 3D viewer — shown only when mesh data is available
            if (meshPositions.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                ResultButton(
                    label   = "VIEW 3D",
                    modifier = Modifier.fillMaxWidth(),
                    color   = Plasma.copy(alpha = 0.75f),
                    onClick = { showViewer = true }
                )
            }

            Spacer(Modifier.height(8.dp))
            ResultButton("CLOSE", Modifier.fillMaxWidth(), Color.White.copy(0.3f), onDismiss)
        }
    }
}

@Composable
fun StatCell(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, fontSize = 20.sp, color = Plasma,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
        Text(label, fontSize = 9.sp, color = Color.White.copy(0.4f),
            fontFamily = FontFamily.Monospace)
    }
}

@Composable
fun ResultButton(label: String, modifier: Modifier, color: Color, onClick: () -> Unit) {
    Box(
        modifier = modifier
            .border(1.dp, color, RoundedCornerShape(8.dp))
            .clip(RoundedCornerShape(8.dp))
            .background(color.copy(alpha = 0.08f))
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(label, fontSize = 11.sp, color = color,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
    }
}
