package com.arhand.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.clickable
import com.arhand.scanner.Scanner

/**
 * Full-screen scan overlay — rendered on top of the AR view during scanning.
 * Shows: pose name, instruction, quality bar, progress rings, countdown.
 */
@Composable
fun ScanOverlay(status: Scanner.ScanStatus, onCancel: () -> Unit) {
    Box(modifier = Modifier.fillMaxSize()) {
        when (status.state) {
            Scanner.ScanState.COUNTDOWN -> CountdownOverlay(status.countdownSec)
            Scanner.ScanState.CAPTURING -> CaptureOverlay(status)
            Scanner.ScanState.PROCESSING -> ProcessingOverlay()
            else -> {}
        }

        // Cancel button — always visible during scan
        if (status.state != Scanner.ScanState.DONE && status.state != Scanner.ScanState.IDLE) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(top = 48.dp, end = 16.dp),
                contentAlignment = Alignment.TopEnd
            ) {
                ScanButton("✕ CANCEL", accentColor = Warn, onClick = onCancel)
            }
        }
    }
}

@Composable
fun CountdownOverlay(sec: Int) {
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val scale by infiniteTransition.animateFloat(
        initialValue = 1f, targetValue = 1.15f,
        animationSpec = infiniteRepeatable(tween(500, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "scale"
    )

    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = sec.toString(),
                fontSize = (64 * scale).sp,
                color = Plasma,
                fontFamily = FontFamily.Monospace,
                fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(12.dp))
            Text("POSITION YOUR HAND", fontSize = 12.sp, color = Color.White.copy(0.7f),
                fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
fun CaptureOverlay(status: Scanner.ScanStatus) {
    Box(Modifier.fillMaxSize()) {
        // Top info bar
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 60.dp)
                .background(UIBg, RoundedCornerShape(12.dp))
                .padding(horizontal = 20.dp, vertical = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "POSE ${status.poseIndex + 1} / 8",
                fontSize = 10.sp, color = Color.White.copy(0.5f),
                fontFamily = FontFamily.Monospace
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = status.pose.label.uppercase(),
                fontSize = 18.sp, color = Plasma,
                fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold
            )
            Spacer(Modifier.height(4.dp))
            Text(
                text = status.pose.instruction,
                fontSize = 12.sp, color = Color.White.copy(0.8f),
                fontFamily = FontFamily.Monospace, textAlign = TextAlign.Center
            )
        }

        // Bottom progress section
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(bottom = 100.dp, start = 24.dp, end = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Quality bar
            Text("QUALITY", fontSize = 9.sp, color = Color.White.copy(0.4f),
                fontFamily = FontFamily.Monospace)
            Spacer(Modifier.height(4.dp))
            LinearProgressIndicator(
                progress = { status.quality },
                modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)),
                color = if (status.quality >= 0.55f) Plasma else Warn,
                trackColor = Color.White.copy(0.1f)
            )
            Spacer(Modifier.height(12.dp))
            // Hold / rep progress
            LinearProgressIndicator(
                progress = { status.holdProgress },
                modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(3.dp)),
                color = Plasma,
                trackColor = Color.White.copy(0.1f)
            )
            Spacer(Modifier.height(8.dp))
            // 8-pose pip row
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                for (i in 0 until 8) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(
                                when {
                                    i < status.poseIndex  -> Plasma
                                    i == status.poseIndex -> Plasma.copy(alpha = 0.6f)
                                    else                  -> Color.White.copy(alpha = 0.15f)
                                }
                            )
                    )
                }
            }
        }
    }
}

@Composable
fun ProcessingOverlay() {
    val infiniteTransition = rememberInfiniteTransition(label = "proc")
    val alpha by infiniteTransition.animateFloat(
        0.4f, 1f,
        infiniteRepeatable(tween(600), RepeatMode.Reverse),
        label = "alpha"
    )
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("BUILDING MODEL", fontSize = 14.sp, color = Plasma.copy(alpha = alpha),
                fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(8.dp))
            Text("Running marching cubes...", fontSize = 11.sp, color = Color.White.copy(0.5f),
                fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
fun ScanButton(label: String, accentColor: Color = Plasma, onClick: () -> Unit) {
    Box(
        modifier = Modifier
            .border(1.dp, accentColor, RoundedCornerShape(8.dp))
            .background(accentColor.copy(0.1f), RoundedCornerShape(8.dp))
            .clip(RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Text(label, fontSize = 11.sp, color = accentColor,
            fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
    }
}
