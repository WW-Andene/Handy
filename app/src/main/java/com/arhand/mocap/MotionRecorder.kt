package com.arhand.mocap

import com.arhand.util.Vec3
import java.io.File
import java.io.FileWriter
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Records timestamped frames of joint rotations and wrist root data into a fixed-size
 * ring buffer (default cap: 60 seconds at 30fps = 1,800 frames), then exports to BVH.
 *
 * Usage:
 * ```kotlin
 * val recorder = MotionRecorder()
 *
 * // Each frame while recording is active:
 * recorder.pushFrame(retargetResult, handIndex = 0)
 *
 * // When the user stops recording:
 * val bvhFile = recorder.exportBvh(outputDir, label = "take_01")
 * recorder.reset()
 * ```
 *
 * Thread safety: [pushFrame] is called from the GL / tracking thread; [exportBvh] and
 * [reset] are called from the UI thread. Synchronization is via [synchronized] on
 * [frames]. Keep the critical section short — only the ring-buffer append/copy.
 */
class MotionRecorder(
    /** Maximum number of frames retained. Oldest frames are dropped when the cap is hit. */
    val maxFrames: Int = MAX_FRAMES_DEFAULT
) {

    companion object {
        const val MAX_FRAMES_DEFAULT = 1_800   // 60 s × 30 fps
        const val FPS = 30
        private const val RAD_TO_DEG = (180.0 / Math.PI).toFloat()

        // BVH channel order for each joint: ZX rotation (Euler ZXY decomposition)
        // Most DCC tools expect ZXY when importing hand motion BVH.
        private val JOINT_NAMES = mapOf(
            BoneRetargeter.JOINT_WRIST      to "Wrist",
            BoneRetargeter.JOINT_THUMB_CMC  to "ThumbCMC",
            BoneRetargeter.JOINT_THUMB_MCP  to "ThumbMCP",
            BoneRetargeter.JOINT_THUMB_IP   to "ThumbIP",
            BoneRetargeter.JOINT_INDEX_MCP  to "IndexMCP",
            BoneRetargeter.JOINT_INDEX_PIP  to "IndexPIP",
            BoneRetargeter.JOINT_INDEX_DIP  to "IndexDIP",
            BoneRetargeter.JOINT_MIDDLE_MCP to "MiddleMCP",
            BoneRetargeter.JOINT_MIDDLE_PIP to "MiddlePIP",
            BoneRetargeter.JOINT_MIDDLE_DIP to "MiddleDIP",
            BoneRetargeter.JOINT_RING_MCP   to "RingMCP",
            BoneRetargeter.JOINT_RING_PIP   to "RingPIP",
            BoneRetargeter.JOINT_RING_DIP   to "RingDIP",
            BoneRetargeter.JOINT_PINKY_MCP  to "PinkyMCP",
            BoneRetargeter.JOINT_PINKY_PIP  to "PinkyPIP",
            BoneRetargeter.JOINT_PINKY_DIP  to "PinkyDIP"
        )

        // Approximate bone lengths (in world-space units) used for BVH OFFSET values.
        // These are the rest-pose distances between joint origins for a normalised hand.
        // They only affect skeleton display in the BVH viewer — animation data is
        // rotation-only so the values are cosmetic.
        private val BONE_LENGTH = mapOf(
            BoneRetargeter.JOINT_WRIST      to 0.00f,   // root — offset from scene origin
            BoneRetargeter.JOINT_THUMB_CMC  to 0.045f,
            BoneRetargeter.JOINT_THUMB_MCP  to 0.030f,
            BoneRetargeter.JOINT_THUMB_IP   to 0.028f,
            BoneRetargeter.JOINT_INDEX_MCP  to 0.070f,
            BoneRetargeter.JOINT_INDEX_PIP  to 0.030f,
            BoneRetargeter.JOINT_INDEX_DIP  to 0.022f,
            BoneRetargeter.JOINT_MIDDLE_MCP to 0.075f,
            BoneRetargeter.JOINT_MIDDLE_PIP to 0.032f,
            BoneRetargeter.JOINT_MIDDLE_DIP to 0.024f,
            BoneRetargeter.JOINT_RING_MCP   to 0.070f,
            BoneRetargeter.JOINT_RING_PIP   to 0.030f,
            BoneRetargeter.JOINT_RING_DIP   to 0.022f,
            BoneRetargeter.JOINT_PINKY_MCP  to 0.055f,
            BoneRetargeter.JOINT_PINKY_PIP  to 0.025f,
            BoneRetargeter.JOINT_PINKY_DIP  to 0.018f
        )

        /** Joint ordering for BVH hierarchy output — wrist first, then finger chains. */
        val HIERARCHY_ORDER: List<Int> = listOf(
            BoneRetargeter.JOINT_WRIST,
            BoneRetargeter.JOINT_THUMB_CMC,
            BoneRetargeter.JOINT_THUMB_MCP,
            BoneRetargeter.JOINT_THUMB_IP,
            BoneRetargeter.JOINT_INDEX_MCP,
            BoneRetargeter.JOINT_INDEX_PIP,
            BoneRetargeter.JOINT_INDEX_DIP,
            BoneRetargeter.JOINT_MIDDLE_MCP,
            BoneRetargeter.JOINT_MIDDLE_PIP,
            BoneRetargeter.JOINT_MIDDLE_DIP,
            BoneRetargeter.JOINT_RING_MCP,
            BoneRetargeter.JOINT_RING_PIP,
            BoneRetargeter.JOINT_RING_DIP,
            BoneRetargeter.JOINT_PINKY_MCP,
            BoneRetargeter.JOINT_PINKY_PIP,
            BoneRetargeter.JOINT_PINKY_DIP
        )
    }

    // ─── Ring buffer ──────────────────────────────────────────────────────────

    /** Immutable snapshot of one recorded frame. */
    data class RecordedFrame(
        /** Elapsed time from recording start, in milliseconds. */
        val timestampMs: Long,
        /** Joint index → quaternion. A frame may have fewer than 16 entries if
         *  retargeting returned partial data (occluded joints). Missing joints
         *  are filled with IDENTITY on export. */
        val jointRotations: Map<Int, Quaternion>,
        /** Wrist world-space root transform for the captured hand. */
        val wristTransform: WristTransform
    )

    private val frames = ArrayDeque<RecordedFrame>(maxFrames)
    private var recordingStartMs: Long = -1L

    // ─── Recording control ────────────────────────────────────────────────────

    /** True while the recorder is actively accepting frames. */
    var isRecording: Boolean = false
        private set

    /** Start a new recording session. Clears any previously buffered frames. */
    fun start() {
        synchronized(frames) {
            frames.clear()
            recordingStartMs = System.currentTimeMillis()
            isRecording = true
        }
    }

    /** Stop accepting new frames. Does not clear buffered data. */
    fun stop() {
        isRecording = false
    }

    /** Stop recording and clear all buffered frames. */
    fun reset() {
        synchronized(frames) {
            isRecording = false
            frames.clear()
            recordingStartMs = -1L
        }
    }

    // ─── Frame ingestion ──────────────────────────────────────────────────────

    /**
     * Push one retargeted frame into the ring buffer.
     *
     * Safe to call from the tracking / GL thread. If the buffer is full the oldest
     * frame is evicted so the most recent [maxFrames] frames are always retained.
     *
     * @param result  Output from [BoneRetargeter.retarget] for the current frame.
     */
    fun pushFrame(result: RetargetResult) {
        if (!isRecording) return
        val now = System.currentTimeMillis()
        val ts  = if (recordingStartMs < 0) 0L else now - recordingStartMs

        val frame = RecordedFrame(
            timestampMs    = ts,
            jointRotations = HashMap(result.jointRotations),   // defensive copy
            wristTransform = result.wristTransform
        )

        synchronized(frames) {
            if (frames.size >= maxFrames) frames.removeFirst()
            frames.addLast(frame)
        }
    }

    // ─── Queries ──────────────────────────────────────────────────────────────

    /** Snapshot of all buffered frames in chronological order. */
    fun frameSnapshot(): List<RecordedFrame> = synchronized(frames) { frames.toList() }

    /** Duration of the buffered recording in milliseconds. 0 if no frames. */
    val durationMs: Long get() = synchronized(frames) {
        if (frames.isEmpty()) 0L else frames.last().timestampMs - frames.first().timestampMs
    }

    /** Number of frames currently buffered. */
    val frameCount: Int get() = synchronized(frames) { frames.size }

    // ─── BVH Export ──────────────────────────────────────────────────────────

    /**
     * Export the current ring buffer as a BVH (Biovision Hierarchy) file.
     *
     * BVH is the universal mocap interchange format — readable by Blender, Maya,
     * MotionBuilder, Unity, and Unreal with no plugin required.
     *
     * Structure:
     * - Root joint: Wrist with 6 channels (Xposition Yposition Zposition Zrotation
     *   Xrotation Yrotation) — root motion enabled.
     * - Finger joints: 3 channels each (Zrotation Xrotation Yrotation) — rotation only.
     * - End sites appended after each leaf joint (required by the BVH spec).
     *
     * Rotation convention: ZXY Euler angles in degrees. This matches the default
     * import settings in Blender and MotionBuilder for hand rigs.
     *
     * @param outputDir  Directory to write the file into (must be writable).
     * @param label      File name stem, e.g. "take_01". Spaces replaced with underscores.
     * @return           The written [File], or null if the buffer is empty or IO fails.
     */
    fun exportBvh(outputDir: File, label: String = "mocap"): File? {
        val snapshot = frameSnapshot()
        if (snapshot.isEmpty()) return null

        val safeName = label.replace(' ', '_').replace(Regex("[^A-Za-z0-9_\\-]"), "")
        val outFile  = File(outputDir, "$safeName.bvh")

        return try {
            FileWriter(outFile).use { w ->
                writeBvhHierarchy(w)
                writeBvhMotion(w, snapshot)
            }
            outFile
        } catch (e: Exception) {
            null
        }
    }

    // ─── BVH Hierarchy section ────────────────────────────────────────────────

    private fun writeBvhHierarchy(w: FileWriter) {
        w.write("HIERARCHY\n")

        // BVH expects a tree structure. Hand topology:
        //   Wrist (root)
        //     ├── ThumbCMC → ThumbMCP → ThumbIP → End
        //     ├── IndexMCP → IndexPIP → IndexDIP → End
        //     ├── MiddleMCP → MiddlePIP → MiddleDIP → End
        //     ├── RingMCP → RingPIP → RingDIP → End
        //     └── PinkyMCP → PinkyPIP → PinkyDIP → End

        w.write("ROOT Wrist\n")
        w.write("{\n")
        w.write("\tOFFSET 0.00 0.00 0.00\n")
        // Root has position + rotation channels for root motion
        w.write("\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")

        writeFingerChain(w, "Thumb",  listOf("ThumbCMC", "ThumbMCP", "ThumbIP"),   depth = 1)
        writeFingerChain(w, "Index",  listOf("IndexMCP", "IndexPIP", "IndexDIP"),   depth = 1)
        writeFingerChain(w, "Middle", listOf("MiddleMCP", "MiddlePIP", "MiddleDIP"), depth = 1)
        writeFingerChain(w, "Ring",   listOf("RingMCP", "RingPIP", "RingDIP"),       depth = 1)
        writeFingerChain(w, "Pinky",  listOf("PinkyMCP", "PinkyPIP", "PinkyDIP"),   depth = 1)

        w.write("}\n")
    }

    private fun writeFingerChain(w: FileWriter, finger: String, joints: List<String>, depth: Int) {
        val indent = "\t".repeat(depth)

        // Map display name back to joint index for bone length lookup
        val jointIndexForName: (String) -> Int? = { name ->
            JOINT_NAMES.entries.firstOrNull { it.value == name }?.key
        }

        joints.forEachIndexed { i, jointName ->
            val jointIdx = jointIndexForName(jointName)
            val len      = BONE_LENGTH[jointIdx] ?: 0.03f
            val offset   = "%.4f %.4f %.4f".format(0f, len, 0f)   // offset along +Y (bone axis)

            w.write("${indent}JOINT $jointName\n")
            w.write("$indent{\n")
            w.write("$indent\tOFFSET $offset\n")
            w.write("$indent\tCHANNELS 3 Zrotation Xrotation Yrotation\n")

            if (i < joints.size - 1) {
                // Recurse into next joint in chain
                writeFingerChain(w, finger, joints.subList(i + 1, joints.size), depth + 1)
            } else {
                // Leaf — append End Site
                val endLen = len * 0.6f
                w.write("$indent\tEnd Site\n")
                w.write("$indent\t{\n")
                w.write("$indent\t\tOFFSET 0.0000 %.4f 0.0000\n".format(endLen))
                w.write("$indent\t}\n")
            }

            w.write("$indent}\n")
            return   // only process first joint; recursion handles the rest
        }
    }

    // ─── BVH Motion section ───────────────────────────────────────────────────

    private fun writeBvhMotion(w: FileWriter, snapshot: List<RecordedFrame>) {
        val frameCount  = snapshot.size
        val frameTime   = 1.0 / FPS   // seconds per frame

        w.write("MOTION\n")
        w.write("Frames: $frameCount\n")
        w.write("Frame Time: %.6f\n".format(frameTime))

        // Channel order must exactly match the HIERARCHY section:
        // Wrist: Xpos Ypos Zpos Zrot Xrot Yrot
        // ThumbCMC: Zrot Xrot Yrot
        // ThumbMCP: Zrot Xrot Yrot
        // ThumbIP: Zrot Xrot Yrot
        // IndexMCP..DIP, MiddleMCP..DIP, RingMCP..DIP, PinkyMCP..DIP: same

        val jointWriteOrder: List<Int> = HIERARCHY_ORDER

        for (frame in snapshot) {
            val sb = StringBuilder()

            for ((idx, jointIdx) in jointWriteOrder.withIndex()) {
                val q   = frame.jointRotations[jointIdx] ?: Quaternion.IDENTITY
                val (ez, ex, ey) = quaternionToZXY(q)

                if (jointIdx == BoneRetargeter.JOINT_WRIST) {
                    // Root: position channels first
                    val pos = frame.wristTransform.position
                    sb.append("%.4f %.4f %.4f ".format(
                        pos.x * 100f,   // BVH convention: centimetres
                        pos.y * 100f,
                        pos.z * 100f
                    ))
                }

                sb.append("%.4f %.4f %.4f".format(ez, ex, ey))
                if (idx < jointWriteOrder.size - 1) sb.append(' ')
            }

            w.write(sb.toString())
            w.write("\n")
        }
    }

    // ─── Euler conversion ─────────────────────────────────────────────────────

    /**
     * Decompose a [Quaternion] into ZXY intrinsic Euler angles (degrees).
     *
     * ZXY is the standard decomposition order used by most BVH importers when
     * reconstructing hand joint rotations from motion capture data.
     *
     * Returns a [Triple] of (Z, X, Y) angles in degrees.
     *
     * Reference: Ken Shoemake, "Euler Angle Conversion", Graphics Gems IV, 1994.
     */
    private fun quaternionToZXY(q: Quaternion): Triple<Float, Float, Float> {
        val x = q.x; val y = q.y; val z = q.z; val w = q.w

        // ZXY decomposition
        val sinX = 2f * (w * x + y * z)
        val cosX = 1f - 2f * (x * x + y * y)

        val sinY_cosX =  2f * (w * y - z * x)
        val cosY_cosX =  1f - 2f * (x * x + z * z)   // Note: not strictly correct for gimbal
                                                       // at X = ±90°; clamp handles boundary.
        val sinZ_cosX =  2f * (w * z + x * y)
        val cosZ_cosX =  1f - 2f * (y * y + z * z)

        val ex = asin(sinX.coerceIn(-1f, 1f))
        val ey = atan2(sinY_cosX, cosY_cosX)
        val ez = atan2(sinZ_cosX, cosZ_cosX)

        return Triple(ez * RAD_TO_DEG, ex * RAD_TO_DEG, ey * RAD_TO_DEG)
    }
}
