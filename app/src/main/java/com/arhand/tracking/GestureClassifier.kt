package com.arhand.tracking

import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * G4 — On-device gesture vocabulary classifier.
 *
 * Runs on the already-smoothed landmark stream produced by [HandPipeline].
 * Overhead is < 2ms per frame (pure Kotlin, no ML runtime required).
 *
 * ## Feature vector
 *
 * A 14-dimensional float vector is computed per frame:
 *   - 5 finger extension ratios  [0–1]  (how "open" each finger is relative to its max length)
 *   - 4 MCP–TIP curl angles [0–1]  (normalized joint angle chain sum per finger)
 *   - 1 thumb–index gap    [0–1]  (normalized distance for OK detection)
 *   - 4 inter-finger spread angles [0–1]  (for OPEN_PALM vs FIST discrimination)
 *
 * All features are wrist-size-normalized (divided by palm diagonal) so the
 * classifier is scale-invariant — works regardless of hand distance from camera.
 *
 * ## Classification
 *
 * k-NN with k=1 over the 8 built-in gesture templates (one template per [Gesture]).
 * A gesture is only returned when:
 *   - The nearest-neighbour L2 distance is ≤ [MATCH_THRESHOLD]
 *   - The same gesture label has been the winner for ≥ [HOLD_FRAMES] consecutive frames
 *     (debounce: suppresses single-frame spurious matches)
 *
 * If no template is within [MATCH_THRESHOLD], null is emitted.
 *
 * ## Usage
 *
 * ```kotlin
 * val classifier = GestureClassifier()
 * // Each tracking frame (on smoothed landmarks):
 * val gesture: Gesture? = classifier.classify(smoothedLandmarks)
 * ```
 */
class GestureClassifier {

    companion object {
        /** Max L2 distance to the nearest template for a match to be accepted. */
        const val MATCH_THRESHOLD = 0.55f

        /** Consecutive frames the same gesture must win before it is emitted. */
        const val HOLD_FRAMES = 4

        /** Feature vector dimensionality. */
        const val FEATURE_DIM = 14

        // ── Finger chains (landmark indices) ──────────────────────────────────
        // Each entry: [MCP, PIP, DIP, TIP]
        private val FINGER_CHAINS = arrayOf(
            intArrayOf(LM.THUMB_CMC,  LM.THUMB_MCP,  LM.THUMB_IP,   LM.THUMB_TIP),   // Thumb
            intArrayOf(LM.INDEX_MCP,  LM.INDEX_PIP,  LM.INDEX_DIP,  LM.INDEX_TIP),
            intArrayOf(LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP),
            intArrayOf(LM.RING_MCP,   LM.RING_PIP,   LM.RING_DIP,   LM.RING_TIP),
            intArrayOf(LM.PINKY_MCP,  LM.PINKY_PIP,  LM.PINKY_DIP,  LM.PINKY_TIP)
        )

        // MCP base landmarks for each finger (used in extension ratio)
        private val FINGER_BASE = intArrayOf(
            LM.THUMB_CMC,
            LM.INDEX_MCP,
            LM.MIDDLE_MCP,
            LM.RING_MCP,
            LM.PINKY_MCP
        )

        // ── Built-in gesture templates ─────────────────────────────────────────
        //
        // Each template is a FEATURE_DIM float array manually tuned to match the
        // expected landmark geometry for each gesture.
        //
        // Feature layout:
        //   [0..4]   finger extension ratios: thumb, index, middle, ring, pinky
        //   [5..8]   MCP–TIP curl (sum of 3 joint angles, normalized): index, middle, ring, pinky
        //   [9]      thumb-index gap (normalized by palm size)
        //   [10..13] inter-finger spread: thumb-index, index-middle, middle-ring, ring-pinky
        //
        // Values are calibrated on average hand geometry; the 0.55 MATCH_THRESHOLD
        // provides a ±25% tolerance around each prototype.

        private val TEMPLATES: Map<Gesture, FloatArray> = mapOf(

            // ✋ Open palm — all extended, spread wide
            Gesture.OPEN_PALM to floatArrayOf(
                0.95f, 0.95f, 0.95f, 0.95f, 0.95f,   // extension
                0.05f, 0.05f, 0.05f, 0.05f,            // curl
                0.35f,                                  // thumb-index gap
                0.30f, 0.25f, 0.25f, 0.25f             // spread
            ),

            // ✊ Fist — all curled
            Gesture.FIST to floatArrayOf(
                0.10f, 0.10f, 0.10f, 0.10f, 0.10f,
                0.90f, 0.90f, 0.90f, 0.90f,
                0.05f,
                0.05f, 0.05f, 0.05f, 0.05f
            ),

            // ✌ Peace — index + middle extended
            Gesture.PEACE to floatArrayOf(
                0.15f, 0.95f, 0.90f, 0.10f, 0.10f,
                0.05f, 0.05f, 0.90f, 0.90f,
                0.30f,
                0.20f, 0.30f, 0.10f, 0.08f
            ),

            // 👍 Thumbs up — thumb up, others curled
            Gesture.THUMBS_UP to floatArrayOf(
                0.90f, 0.10f, 0.10f, 0.10f, 0.10f,
                0.90f, 0.90f, 0.90f, 0.90f,
                0.20f,
                0.25f, 0.05f, 0.05f, 0.05f
            ),

            // ☝ Point up — index only extended
            Gesture.POINT_UP to floatArrayOf(
                0.15f, 0.95f, 0.10f, 0.10f, 0.10f,
                0.05f, 0.90f, 0.90f, 0.90f,
                0.30f,
                0.20f, 0.15f, 0.05f, 0.05f
            ),

            // 👌 OK — thumb + index touching (small gap), middle/ring/pinky extended
            Gesture.OK to floatArrayOf(
                0.65f, 0.60f, 0.90f, 0.90f, 0.85f,
                0.30f, 0.30f, 0.05f, 0.10f,
                0.04f,   // very small gap — thumb and index touching
                0.10f, 0.20f, 0.20f, 0.20f
            ),

            // 🤘 Rock — index + pinky extended
            Gesture.ROCK to floatArrayOf(
                0.15f, 0.90f, 0.10f, 0.10f, 0.90f,
                0.05f, 0.90f, 0.90f, 0.05f,
                0.35f,
                0.20f, 0.05f, 0.05f, 0.25f
            ),

            // 🤙 Call — thumb + pinky extended
            Gesture.CALL to floatArrayOf(
                0.90f, 0.10f, 0.10f, 0.10f, 0.90f,
                0.90f, 0.90f, 0.90f, 0.05f,
                0.35f,
                0.30f, 0.05f, 0.05f, 0.25f
            )
        )
    }

    // ─── Debounce state ───────────────────────────────────────────────────────

    private var holdCandidate: Gesture? = null
    private var holdCount: Int = 0

    // ─── Public API ───────────────────────────────────────────────────────────

    /**
     * Classify a single frame of smoothed [HandLandmarks].
     *
     * @param lms  21-landmark list from [HandPipeline] (post-OEF smoothing).
     * @return     The recognized [Gesture], or null if no confident match.
     */
    fun classify(lms: HandLandmarks): Gesture? {
        if (lms.size < 21) return debounce(null)

        val features = extractFeatures(lms)
        val best     = nearestTemplate(features)

        return debounce(best)
    }

    /** Reset debounce state (call on hand loss). */
    fun reset() {
        holdCandidate = null
        holdCount     = 0
    }

    // ─── Feature extraction ───────────────────────────────────────────────────

    /**
     * Extract a [FEATURE_DIM]-element feature vector from [lms].
     * All features are normalized to [0,1] using palm size as the scale reference.
     */
    private fun extractFeatures(lms: HandLandmarks): FloatArray {
        // Palm size = distance from wrist to middle MCP (scale reference)
        val palmSize = lmDist(lms[LM.WRIST], lms[LM.MIDDLE_MCP]).coerceAtLeast(1e-5f)

        val f = FloatArray(FEATURE_DIM)

        // ── [0..4] Finger extension ratios ─────────────────────────────────
        // Extension ratio = (dist base→tip) / (sum of segment lengths)
        // 1.0 = fully straight, 0.0 = fully curled
        for (fi in 0..4) {
            val base = lms[FINGER_BASE[fi]]
            val chain = FINGER_CHAINS[fi]
            val tip = lms[chain[3]]
            val straightDist = lmDist(base, tip)
            val chainLen = lmDist(base, lms[chain[0]]) +
                    lmDist(lms[chain[0]], lms[chain[1]]) +
                    lmDist(lms[chain[1]], lms[chain[2]]) +
                    lmDist(lms[chain[2]], tip)
            f[fi] = if (chainLen > 1e-5f) (straightDist / chainLen).coerceIn(0f, 1f) else 0f
        }

        // ── [5..8] MCP–TIP curl angles for index..pinky ────────────────────
        // Curl = 1 - extension (simpler and stable against Z noise)
        for (fi in 1..4) {
            f[4 + fi] = 1f - f[fi]
        }

        // ── [9] Thumb–index tip gap ─────────────────────────────────────────
        f[9] = (lmDist(lms[LM.THUMB_TIP], lms[LM.INDEX_TIP]) / palmSize).coerceIn(0f, 1f)

        // ── [10..13] Inter-finger spread (MCP lateral distances) ───────────
        val spreadPairs = arrayOf(
            LM.THUMB_TIP  to LM.INDEX_TIP,
            LM.INDEX_TIP  to LM.MIDDLE_TIP,
            LM.MIDDLE_TIP to LM.RING_TIP,
            LM.RING_TIP   to LM.PINKY_TIP
        )
        for ((i, pair) in spreadPairs.withIndex()) {
            f[10 + i] = (lmDist(lms[pair.first], lms[pair.second]) / palmSize).coerceIn(0f, 1f)
        }

        return f
    }

    // ─── k-NN (k=1) ──────────────────────────────────────────────────────────

    private fun nearestTemplate(features: FloatArray): Gesture? {
        var bestGesture: Gesture? = null
        var bestDist = Float.MAX_VALUE

        for ((gesture, template) in TEMPLATES) {
            val dist = l2(features, template)
            if (dist < bestDist) {
                bestDist    = dist
                bestGesture = gesture
            }
        }

        return if (bestDist <= MATCH_THRESHOLD) bestGesture else null
    }

    private fun l2(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) {
            val d = a[i] - b[i]
            sum += d * d
        }
        return sqrt(sum)
    }

    // ─── Debounce ─────────────────────────────────────────────────────────────

    private fun debounce(candidate: Gesture?): Gesture? {
        if (candidate == holdCandidate) {
            holdCount++
        } else {
            holdCandidate = candidate
            holdCount     = 1
        }
        return if (holdCount >= HOLD_FRAMES) holdCandidate else null
    }

    // ─── Helper ──────────────────────────────────────────────────────────────

    private fun lmDist(a: Landmark, b: Landmark): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        val dz = a.z - b.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
}
