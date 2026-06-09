package com.arhand.scanner

import com.arhand.tracking.HandLandmarks
import com.arhand.tracking.LM
import com.arhand.tracking.Landmark
import com.arhand.util.Vec3
import com.arhand.util.landmarkToWorld
import kotlin.math.PI
import kotlin.math.sqrt

/**
 * Measured dimensions derived from the 21-landmark hand skeleton.
 *
 * All values are in world-space units (same coordinate system as the SDF carver).
 * Conversion to mm requires a scale calibration step — see [HandBiometrics.WORLD_TO_MM].
 *
 * Use cases: ring sizing, glove sizing, VR avatar calibration.
 */
data class HandBiometrics(
    /** Straight-line distance between INDEX_MCP (5) and PINKY_MCP (17) — knuckle row. */
    val palmWidth: Float,

    /** Sum of 3 segment lengths for each finger: MCP→PIP, PIP→DIP, DIP→TIP. */
    val thumbLength: Float,
    val indexLength: Float,
    val middleLength: Float,
    val ringLength: Float,
    val pinkyLength: Float,

    /** Distance THUMB_TIP (4) → PINKY_TIP (20) — maximum spread. */
    val handSpan: Float,

    /**
     * Knuckle breadth: INDEX_MCP (5) → PINKY_MCP (17) with a +12% anatomical correction
     * to account for the soft tissue and first metacarpal not captured in the skeleton.
     */
    val knuckleBreadth: Float,

    /**
     * Wrist circumference proxy: π × wrist_diameter, where wrist_diameter is estimated
     * as 1.4× the WRIST (0) → INDEX_MCP (5) segment length — an empirically derived ratio
     * consistent with adult hand proportions.
     */
    val wristCircumference: Float,

    /** Largest single-pose quality score used when computing these metrics. */
    val sourcePoseQuality: Float
) {
    companion object {
        /**
         * Approximate world-unit → mm scale factor.
         *
         * Derived from the VIEW_SCALE = 0.65 and typical MediaPipe depth range.
         * A palm width of ~0.10 world units ≈ 80 mm for an average adult hand.
         * This gives a factor of ~800.
         *
         * For high-accuracy use, expose a calibration flow where the user places their
         * hand against a known-size reference (e.g. a credit card, 85.6×54mm).
         */
        const val WORLD_TO_MM = 800f

        /**
         * Compute biometrics from the best-quality landmarks accumulated during a scan.
         *
         * @param frames List of (landmarks, qualityScore) pairs — one entry per scan pose.
         *               Uses the frame with the highest quality score.
         * @param aspect The device aspect ratio used when these landmarks were captured.
         */
        fun compute(
            frames: List<Pair<HandLandmarks, Float>>,
            aspect: Float
        ): HandBiometrics? {
            if (frames.isEmpty()) return null

            // Pick the highest-quality frame
            val (lms, quality) = frames.maxByOrNull { it.second } ?: return null
            if (lms.size < 21) return null

            // Convert all 21 landmarks to world-space Vec3
            val world: List<Vec3> = lms.map { lm ->
                val (wx, wy, wz) = landmarkToWorld(lm, aspect, mirrorX = false)
                Vec3(wx, wy, wz)
            }

            fun dist(a: Int, b: Int): Float = (world[a] - world[b]).length()

            // Palm width: INDEX_MCP → PINKY_MCP
            val palmWidth = dist(LM.INDEX_MCP, LM.PINKY_MCP)

            // Finger lengths: sum of 3 phalanx segments each
            val thumbLength  = dist(LM.THUMB_CMC,  LM.THUMB_MCP)  +
                               dist(LM.THUMB_MCP,  LM.THUMB_IP)   +
                               dist(LM.THUMB_IP,   LM.THUMB_TIP)

            val indexLength  = dist(LM.INDEX_MCP,  LM.INDEX_PIP)  +
                               dist(LM.INDEX_PIP,  LM.INDEX_DIP)  +
                               dist(LM.INDEX_DIP,  LM.INDEX_TIP)

            val middleLength = dist(LM.MIDDLE_MCP, LM.MIDDLE_PIP) +
                               dist(LM.MIDDLE_PIP, LM.MIDDLE_DIP) +
                               dist(LM.MIDDLE_DIP, LM.MIDDLE_TIP)

            val ringLength   = dist(LM.RING_MCP,   LM.RING_PIP)   +
                               dist(LM.RING_PIP,   LM.RING_DIP)   +
                               dist(LM.RING_DIP,   LM.RING_TIP)

            val pinkyLength  = dist(LM.PINKY_MCP,  LM.PINKY_PIP)  +
                               dist(LM.PINKY_PIP,  LM.PINKY_DIP)  +
                               dist(LM.PINKY_DIP,  LM.PINKY_TIP)

            // Hand span: thumb tip → pinky tip
            val handSpan = dist(LM.THUMB_TIP, LM.PINKY_TIP)

            // Knuckle breadth: INDEX_MCP → PINKY_MCP + 12% soft-tissue correction
            val knuckleBreadth = palmWidth * 1.12f

            // Wrist circumference proxy: π × (1.4 × wrist→index_mcp)
            val wristDiameter  = dist(LM.WRIST, LM.INDEX_MCP) * 1.4f
            val wristCircumference = (PI * wristDiameter).toFloat()

            return HandBiometrics(
                palmWidth         = palmWidth,
                thumbLength       = thumbLength,
                indexLength       = indexLength,
                middleLength      = middleLength,
                ringLength        = ringLength,
                pinkyLength       = pinkyLength,
                handSpan          = handSpan,
                knuckleBreadth    = knuckleBreadth,
                wristCircumference = wristCircumference,
                sourcePoseQuality = quality
            )
        }

        /** Convert a world-space measurement to millimetres. */
        fun toMm(worldUnits: Float): Float = worldUnits * WORLD_TO_MM

        /** Format a world-space value as "XX.X mm" for display. */
        fun fmtMm(worldUnits: Float): String = "%.1f mm".format(toMm(worldUnits))
    }

    /** Serialise to a compact JSON string for DataStore persistence. */
    fun toJson(): String = """
        {
          "palmWidth":$palmWidth,
          "thumbLength":$thumbLength,
          "indexLength":$indexLength,
          "middleLength":$middleLength,
          "ringLength":$ringLength,
          "pinkyLength":$pinkyLength,
          "handSpan":$handSpan,
          "knuckleBreadth":$knuckleBreadth,
          "wristCircumference":$wristCircumference,
          "sourcePoseQuality":$sourcePoseQuality
        }
    """.trimIndent()

    companion object {
        /** Deserialise from a toJson() string. Returns null on any parse failure. */
        fun fromJson(json: String): HandBiometrics? = try {
            fun extract(key: String): Float {
                val regex = Regex(""""$key"\s*:\s*([\d.Ee+-]+)""")
                return regex.find(json)?.groupValues?.get(1)?.toFloat()
                    ?: error("missing key $key")
            }
            HandBiometrics(
                palmWidth          = extract("palmWidth"),
                thumbLength        = extract("thumbLength"),
                indexLength        = extract("indexLength"),
                middleLength       = extract("middleLength"),
                ringLength         = extract("ringLength"),
                pinkyLength        = extract("pinkyLength"),
                handSpan           = extract("handSpan"),
                knuckleBreadth     = extract("knuckleBreadth"),
                wristCircumference = extract("wristCircumference"),
                sourcePoseQuality  = extract("sourcePoseQuality")
            )
        } catch (_: Exception) { null }
    }
}
