package com.arhand.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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

val UIBg = Color(0xD20A0D12)
val Plasma = Color(0xFF7B2FBE)
val Warn   = Color(0xFFFF5722)
val RecRed = Color(0xFFE53935)

/**
 * Bottom control panel — matches the HTML prototype's control bar.
 * Buttons: Mode | Torch | Cloud | Camera | REC | Scan
 */
@Composable
fun ControlPanel(
    renderMode: String,
    torchOn: Boolean,
    showCloud: Boolean,
    scanActive: Boolean,
    hasModel: Boolean,
    isRecording: Boolean,
    onMode: () -> Unit,
    onTorch: () -> Unit,
    onCloud: () -> Unit,
    onCamera: () -> Unit,
    onScan: () -> Unit,
    onViewModel: () -> Unit,
    onRecord: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(UIBg)
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            CtrlButton("MODE\n$renderMode", active = true, onClick = onMode)
            CtrlButton("TORCH\n${if(torchOn) "ON" else "OFF"}", active = torchOn, onClick = onTorch)
            CtrlButton("CLOUD\n${if(showCloud) "ON" else "OFF"}", active = showCloud, onClick = onCloud)
            CtrlButton("FLIP\nCAM", active = false, onClick = onCamera)

            // REC button — red when recording, dim when idle
            CtrlButton(
                label       = if (isRecording) "STOP\nREC" else "REC",
                active      = isRecording,
                accentColor = RecRed,
                onClick     = onRecord
            )

            if (hasModel) {
                CtrlButton("3D\nMODEL", active = true, onClick = onViewModel)
            } else {
                CtrlButton(if (scanActive) "CANCEL\nSCAN" else "START\nSCAN",
                    active = !scanActive,
                    accentColor = if (scanActive) Warn else Plasma,
                    onClick = onScan
                )
            }
        }
    }
}

@Composable
fun CtrlButton(
    label: String,
    active: Boolean,
    accentColor: Color = Plasma,
    onClick: () -> Unit
) {
    val borderColor = if (active) accentColor else Color.White.copy(alpha = 0.15f)
    val textColor   = if (active) accentColor else Color.White.copy(alpha = 0.5f)

    Column(
        modifier = Modifier
            .width(60.dp)
            .clip(RoundedCornerShape(8.dp))
            .border(1.dp, borderColor, RoundedCornerShape(8.dp))
            .background(if (active) accentColor.copy(alpha = 0.07f) else Color.Transparent)
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp, horizontal = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        label.split("\n").forEach { line ->
            Text(
                text = line,
                fontSize = 9.sp,
                color = textColor,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
