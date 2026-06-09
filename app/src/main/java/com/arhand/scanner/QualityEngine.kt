package com.arhand.scanner

import com.arhand.tracking.HandLandmarks
import com.arhand.tracking.LM
import com.arhand.util.lmDist
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Quality gating engine — port of SCANNER._qualityGate() from the HTML prototype.
 *
 * Computes a composite quality score [0,1] from:
 *   1. Spread score — how far apart the fingers are (normalized)
 *   2. Depth variance — how much Z variation exists (proxy for 3D pose info)
 *   3. Position score — hand is centered and not at extremes
 *
 * Also evaluates motion completions (fingerCurl, wristRoll, fingerWave, thumbWave).
 */
object QualityEngine {

    const val QUALITY_THRESHOLD = 0.55f

    data class QualityResult(
        val score: Float,
        val spreadScore: Float,
        val depthVariance: Float,
        val positionScore: Float,
        val passed: Boolean
    )

    fun evaluate(lms: HandLandmarks, claheContrast: Float = 1f): QualityResult {
        if (lms.size < 21) return QualityResult(0f, 0f, 0f, 0f, false)

        // 1. Spread score: average tip-to-tip distance normalized by palm size
        val palmSize = lmDist(lms[LM.WRIST], lms[LM.MIDDLE_MCP]).coerceAtLeast(0.001f)
        val tips = listOf(LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)
        var spreadSum = 0f
        var spreadCount = 0
        for (i in tips.indices) {
            for (j in i + 1 until tips.size) {
                spreadSum += lmDist(lms[tips[i]], lms[tips[j]])
                spreadCount++
            }
        }
        val avgSpread = if (spreadCount > 0) spreadSum / spreadCount else 0f
        val spreadScore = (avgSpread / (palmSize * 1.8f)).coerceIn(0f, 1f)

        // 2. Depth variance
        val zVals = lms.map { it.z }
        val zMean = zVals.average().toFloat()
        val zVar  = zVals.map { (it - zMean) * (it - zMean) }.average().toFloat()
        val depthVariance = (zVar * 200f).coerceIn(0f, 1f)

        // 3. Position score — center of hand should be near (0.5, 0.5) in normalized coords
        val cx = lms.map { it.x }.average().toFloat()
        val cy = lms.map { it.y }.average().toFloat()
        val distFromCenter = sqrt((cx - 0.5f) * (cx - 0.5f) + (cy - 0.5f) * (cy - 0.5f))
        val positionScore = (1f - distFromCenter * 2.5f).coerceIn(0f, 1f)

        // Composite — weights match HTML prototype
        val score = (spreadScore * 0.35f +
                     depthVariance * 0.25f +
                     positionScore * 0.25f +
                     (claheContrast.coerceIn(0f, 1f)) * 0.15f)

        return QualityResult(
            score = score,
            spreadScore = spreadScore,
            depthVariance = depthVariance,
            positionScore = positionScore,
            passed = score >= QUALITY_THRESHOLD
        )
    }

    // ─── Motion detection ─────────────────────────────────────────────────

    /**
     * Detect fingerCurl: average distance of fingertips to palm center.
     * Returns 0..1 where 1 = fully open, 0 = fully closed.
     */
    fun fingerCurlScore(lms: HandLandmarks): Float {
        if (lms.size < 21) return 0f
        val palmCenter = lms[LM.MIDDLE_MCP]
        val tips = listOf(LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)
        val palmSize = lmDist(lms[LM.WRIST], lms[LM.MIDDLE_MCP]).coerceAtLeast(0.001f)
        val avgDist = tips.map { lmDist(lms[it], palmCenter) }.average().toFloat()
        return (avgDist / (palmSize * 1.5f)).coerceIn(0f, 1f)
    }

    /**
     * Detect wristRoll: lateral displacement of wrist relative to index/pinky MCPs.
     * Returns angle proxy (-1..1) where sign changes on roll reversal.
     */
    fun wristRollScore(lms: HandLandmarks): Float {
        if (lms.size < 18) return 0f
        val wrist = lms[LM.WRIST]
        val indexMcp = lms[LM.INDEX_MCP]
        val pinkyMcp = lms[LM.PINKY_MCP]
        val palmMidX = (indexMcp.x + pinkyMcp.x) / 2f
        return (wrist.x - palmMidX) * 5f
    }

    /**
     * Detect fingerWave: sequential spread between adjacent fingertips.
     */
    fun fingerWaveScore(lms: HandLandmarks): Float {
        if (lms.size < 21) return 0f
        val tips = listOf(LM.INDEX_TIP, LM.MIDDLE_TIP, LM.RING_TIP, LM.PINKY_TIP)
        var diff = 0f
        for (i in 0 until tips.size - 1) {
            diff += abs(lms[tips[i]].y - lms[tips[i + 1]].y)
        }
        return (diff * 8f).coerceIn(0f, 1f)
    }

    /**
     * Detect thumbWave: thumb tip distance from index finger base.
     */
    fun thumbWaveScore(lms: HandLandmarks): Float {
        if (lms.size < 9) return 0f
        val palmSize = lmDist(lms[LM.WRIST], lms[LM.MIDDLE_MCP]).coerceAtLeast(0.001f)
        val dist = lmDist(lms[LM.THUMB_TIP], lms[LM.INDEX_MCP])
        return (dist / (palmSize * 1.2f)).coerceIn(0f, 1f)
    }
}
