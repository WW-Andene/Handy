package com.arhand.scanner

import com.arhand.tracking.HandLandmarks
import com.arhand.util.Vec3
import com.arhand.util.landmarkToWorld
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/**
 * 8-pose scan state machine — port of SCANNER from the HTML prototype.
 *
 * States:
 *   IDLE → COUNTDOWN → CAPTURING → PROCESSING → DONE | FAILED
 *
 * Flow per pose:
 *   1. QualityEngine gates: must pass QUALITY_THRESHOLD
 *   2. Static pose: hold for holdMs at sufficient quality
 *   3. Motion pose: detect reps using motion score crossing threshold
 *   4. On completion: capture point cloud snapshot and advance to next pose
 */
class Scanner {

    enum class ScanState { IDLE, COUNTDOWN, CAPTURING, PROCESSING, DONE, FAILED }

    data class ScanStatus(
        val state: ScanState = ScanState.IDLE,
        val poseIndex: Int = 0,
        val pose: ScanPose = ScanPoses.ALL[0],
        val quality: Float = 0f,
        val holdProgress: Float = 0f,   // 0..1 for static hold progress bar
        val repCount: Int = 0,          // current rep count for motion poses
        val countdownSec: Int = 3,
        val totalPoints: Int = 0,
        val poseScores: List<Float> = emptyList()
    )

    private val _status = MutableStateFlow(ScanStatus())
    val status: StateFlow<ScanStatus> = _status

    private var poseIndex   = 0
    private var scanState   = ScanState.IDLE
    private var holdStartMs = 0L
    private var countdownStartMs = 0L

    // Accumulated point cloud across all poses
    val cloudPoints = mutableListOf<Vec3>()
    private val poseScores = mutableListOf<Float>()

    // Motion state
    private var lastMotionScore = 0f
    private var repCount        = 0
    private var motionPhase     = false   // true = moving in "open" direction

    // CLAHE
    private val clahe = CLAHEAnalyzer()

    fun start() {
        poseIndex = 0
        scanState = ScanState.COUNTDOWN
        countdownStartMs = System.currentTimeMillis()
        cloudPoints.clear()
        poseScores.clear()
        repCount = 0
        emitStatus()
    }

    fun reset() {
        poseIndex = 0
        scanState = ScanState.IDLE
        cloudPoints.clear()
        poseScores.clear()
        repCount = 0
        holdStartMs = 0L
        emitStatus()
    }

    /**
     * Feed a processed hand frame.
     * Call every tracking frame while scan is active.
     */
    fun update(lms: HandLandmarks?, claheContrast: Float, nowMs: Long, aspect: Float) {
        when (scanState) {
            ScanState.COUNTDOWN -> updateCountdown(nowMs)
            ScanState.CAPTURING -> updateCapture(lms, claheContrast, nowMs, aspect)
            else -> {}
        }
    }

    private fun updateCountdown(nowMs: Long) {
        val elapsed = (nowMs - countdownStartMs) / 1000f
        val remaining = (3f - elapsed).coerceAtLeast(0f).toInt() + 1
        _status.value = _status.value.copy(
            state = ScanState.COUNTDOWN,
            countdownSec = remaining
        )
        if (elapsed >= 3f) {
            scanState = ScanState.CAPTURING
            holdStartMs = nowMs
            emitStatus()
        }
    }

    private fun updateCapture(lms: HandLandmarks?, claheContrast: Float, nowMs: Long, aspect: Float) {
        if (lms == null) {
            emitStatus(); return
        }

        val pose = ScanPoses.ALL[poseIndex]
        val quality = QualityEngine.evaluate(lms, claheContrast)

        var holdProgress = 0f
        var advanced = false

        if (quality.passed) {
            when (pose.type) {
                PoseType.STATIC -> {
                    if (holdStartMs == 0L) holdStartMs = nowMs
                    val held = nowMs - holdStartMs
                    holdProgress = (held.toFloat() / pose.holdMs).coerceIn(0f, 1f)
                    if (held >= pose.holdMs) advanced = true
                }
                PoseType.MOTION -> {
                    val score = when (pose.motionKey) {
                        "fingerCurl"  -> QualityEngine.fingerCurlScore(lms)
                        "wristRoll"   -> QualityEngine.wristRollScore(lms)
                        "fingerWave"  -> QualityEngine.fingerWaveScore(lms)
                        "thumbWave"   -> QualityEngine.thumbWaveScore(lms)
                        else          -> 0f
                    }
                    // Rep detection: crossing 0.5 threshold in each direction
                    if (!motionPhase && score > 0.65f) {
                        motionPhase = true
                    } else if (motionPhase && score < 0.35f) {
                        motionPhase = false
                        repCount++
                    }
                    holdProgress = (repCount.toFloat() / pose.reps).coerceIn(0f, 1f)
                    if (repCount >= pose.reps) advanced = true
                }
            }
        } else {
            holdStartMs = 0L
        }

        if (advanced) {
            // Capture snapshot points for this pose
            capturePosePoints(lms, aspect, quality.score)
            advancePose()
        } else {
            _status.value = _status.value.copy(
                state = ScanState.CAPTURING,
                poseIndex = poseIndex,
                pose = pose,
                quality = quality.score,
                holdProgress = holdProgress,
                repCount = repCount
            )
        }
    }

    private fun capturePosePoints(lms: HandLandmarks, aspect: Float, qualityScore: Float) {
        // Sample 60 points per pose — mirrors SCANNER._capturePoints() from HTML
        // Interpolate between landmark pairs with slight random jitter
        val pairs = com.arhand.tracking.CONNECTIONS
        for ((a, b) in pairs) {
            if (a >= lms.size || b >= lms.size) continue
            val pA = lms[a]; val pB = lms[b]
            for (t in 0..3) {
                val tv = t / 3f
                val x = pA.x + (pB.x - pA.x) * tv
                val y = pA.y + (pB.y - pA.y) * tv
                val z = pA.z + (pB.z - pA.z) * tv
                val (wx, wy, wz) = landmarkToWorld(
                    com.arhand.tracking.Landmark(x, y, z), aspect, mirrorX = false
                )
                cloudPoints.add(Vec3(wx, wy, wz))
            }
        }
        poseScores.add(qualityScore)
    }

    private fun advancePose() {
        holdStartMs = 0L
        repCount = 0
        motionPhase = false
        poseIndex++

        if (poseIndex >= ScanPoses.COUNT) {
            scanState = ScanState.PROCESSING
            _status.value = _status.value.copy(
                state = ScanState.PROCESSING,
                totalPoints = cloudPoints.size,
                poseScores = poseScores.toList()
            )
        } else {
            scanState = ScanState.CAPTURING
            emitStatus()
        }
    }

    fun markDone() {
        scanState = ScanState.DONE
        _status.value = _status.value.copy(state = ScanState.DONE, totalPoints = cloudPoints.size)
    }

    fun markFailed() {
        scanState = ScanState.FAILED
        _status.value = _status.value.copy(state = ScanState.FAILED)
    }

    private fun emitStatus() {
        val pose = ScanPoses.ALL.getOrElse(poseIndex) { ScanPoses.ALL.last() }
        _status.value = _status.value.copy(
            state = scanState,
            poseIndex = poseIndex,
            pose = pose,
            totalPoints = cloudPoints.size,
            poseScores = poseScores.toList()
        )
    }
}
