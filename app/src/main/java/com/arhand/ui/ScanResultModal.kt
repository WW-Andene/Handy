package com.arhand.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
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
import com.arhand.scanner.HandBiometrics

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
    biometrics: HandBiometrics?,
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
                .padding(24.dp)
                .verticalScroll(rememberScrollState()),
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

            // Hand biometrics panel
            if (biometrics != null) {
                BiometricsPanel(biometrics)
                Spacer(Modifier.height(20.dp))
            }

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

// ─────────────────────────────────────────────────────────────────────────────
// Hand Biometrics Panel
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Displays the computed hand biometrics in a compact monospace grid.
 * All values are shown in mm using [HandBiometrics.fmtMm].
 */
@Composable
fun BiometricsPanel(b: HandBiometrics) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, Plasma.copy(alpha = 0.2f), RoundedCornerShape(10.dp))
            .clip(RoundedCornerShape(10.dp))
            .background(Color.White.copy(alpha = 0.03f))
            .padding(horizontal = 14.dp, vertical = 12.dp)
    ) {
        // Section header
        Text(
            text = "HAND BIOMETRICS",
            fontSize = 9.sp,
            color = Plasma.copy(alpha = 0.7f),
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.5.sp
        )

        Spacer(Modifier.height(10.dp))

        // Top row: palm width, hand span, wrist
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            BiometricCell("PALM WIDTH",  HandBiometrics.fmtMm(b.palmWidth))
            BiometricCell("HAND SPAN",   HandBiometrics.fmtMm(b.handSpan))
            BiometricCell("WRIST CIRC",  HandBiometrics.fmtMm(b.wristCircumference))
        }

        Spacer(Modifier.height(10.dp))

        // Divider
        Box(
            Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color.White.copy(alpha = 0.07f))
        )

        Spacer(Modifier.height(10.dp))

        // Finger lengths row
        Text(
            text = "FINGER LENGTHS",
            fontSize = 8.sp,
            color = Color.White.copy(alpha = 0.35f),
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.sp
        )

        Spacer(Modifier.height(6.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly
        ) {
            FingerLengthCell("THB", b.thumbLength)
            FingerLengthCell("IDX", b.indexLength)
            FingerLengthCell("MID", b.middleLength)
            FingerLengthCell("RNG", b.ringLength)
            FingerLengthCell("PNK", b.pinkyLength)
        }

        Spacer(Modifier.height(8.dp))

        // Inline finger-length bar chart — relative proportions
        val lengths = listOf(b.thumbLength, b.indexLength, b.middleLength, b.ringLength, b.pinkyLength)
        val maxLen = lengths.max()
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(3.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            lengths.forEach { len ->
                val fraction = if (maxLen > 0f) len / maxLen else 0f
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .height((28 * fraction).dp)
                        .clip(RoundedCornerShape(topStart = 2.dp, topEnd = 2.dp))
                        .background(Plasma.copy(alpha = 0.55f + 0.35f * fraction))
                )
            }
        }

        Spacer(Modifier.height(10.dp))

        // Divider
        Box(
            Modifier
                .fillMaxWidth()
                .height(1.dp)
                .background(Color.White.copy(alpha = 0.07f))
        )

        Spacer(Modifier.height(8.dp))

        // Knuckle breadth + quality note
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            BiometricCell("KNUCKLE BREADTH", HandBiometrics.fmtMm(b.knuckleBreadth))
            Text(
                text = "QUALITY %.0f%%".format(b.sourcePoseQuality * 100f),
                fontSize = 8.sp,
                color = Plasma.copy(alpha = 0.45f),
                fontFamily = FontFamily.Monospace
            )
        }

        Spacer(Modifier.height(6.dp))

        // Use-case hint
        Text(
            text = "Use for ring sizing · glove sizing · VR avatar calibration",
            fontSize = 8.sp,
            color = Color.White.copy(alpha = 0.25f),
            fontFamily = FontFamily.Monospace
        )
    }
}

@Composable
private fun BiometricCell(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = value,
            fontSize = 14.sp,
            color = Color.White.copy(alpha = 0.9f),
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            fontSize = 7.sp,
            color = Color.White.copy(alpha = 0.35f),
            fontFamily = FontFamily.Monospace,
            letterSpacing = 0.8.sp
        )
    }
}

@Composable
private fun FingerLengthCell(abbrev: String, worldLen: Float) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = HandBiometrics.fmtMm(worldLen),
            fontSize = 9.sp,
            color = Color.White.copy(alpha = 0.75f),
            fontFamily = FontFamily.Monospace
        )
        Text(
            text = abbrev,
            fontSize = 7.sp,
            color = Color.White.copy(alpha = 0.3f),
            fontFamily = FontFamily.Monospace
        )
    }
}
