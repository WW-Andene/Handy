package com.arhand.mocap

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * Streams live hand joint rotations as OSC (Open Sound Control) messages over UDP.
 *
 * OSC is the standard protocol used by professional mocap systems (Vicon, OptiTrack)
 * and is natively supported by Unreal Engine, Unity (via OSCJack / extOSC), TouchDesigner,
 * and Notch. This allows Handy to drive a desktop character rig in real time without a
 * USB connection or file export step.
 *
 * ## Message format
 *
 * Each [sendFrame] call emits **one bundle** containing 17 OSC messages:
 *
 * | Address                        | Arguments              |
 * |--------------------------------|------------------------|
 * | `/hand/wrist/position`         | float x, float y, float z |
 * | `/hand/<JointName>/rotation`   | float x, float y, float z, float w (quaternion) |
 *
 * Joint names match [BoneRetargeter] constants (e.g. `/hand/IndexMCP/rotation`).
 * Addresses and types are fully compatible with the OSC 1.0 specification.
 *
 * ## Bundle structure
 *
 * All per-frame messages are packed into a single OSC bundle with the timetag
 * set to the OSC "immediate" sentinel (`0x0000000000000001`). This lets receiving
 * tools (TouchDesigner, Unreal) process the whole frame atomically rather than
 * interleaving partial frames.
 *
 * ## Target latency
 *
 * Packet construction and send happen on a dedicated coroutine with a
 * [Channel.CONFLATED] queue — if the send coroutine falls behind the tracking
 * thread (e.g. OS scheduler jitter), the latest frame replaces the queued one
 * instead of building up a backlog. Over a local WiFi network (5 GHz) the
 * round-trip is typically under 2ms; one-way under 1ms.
 *
 * ## Usage
 * ```kotlin
 * val streamer = OscStreamer()
 * streamer.start(host = "192.168.1.42", port = 9000)
 *
 * // Each tracking frame:
 * streamer.sendFrame(retargetResult)
 *
 * // On teardown:
 * streamer.stop()
 * ```
 *
 * Thread safety: [sendFrame] is safe to call from any thread. [start] and [stop]
 * must be called from the same thread (typically the ViewModel/main thread).
 */
class OscStreamer {

    companion object {
        const val DEFAULT_PORT = 9000
        const val DEFAULT_HOST = "255.255.255.255"   // LAN broadcast

        // OSC bundle header: "#bundle\0" + timetag (immediate = 1)
        private val BUNDLE_HEADER: ByteArray = run {
            val buf = ByteBuffer.allocate(16).order(ByteOrder.BIG_ENDIAN)
            buf.put("#bundle\u0000".toByteArray(StandardCharsets.US_ASCII))
            buf.putInt(0)            // timetag seconds (immediate)
            buf.putInt(1)            // timetag fraction (immediate sentinel)
            buf.array()
        }

        // Joint name table for OSC address construction
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

        // Pre-build type tag strings (immutable — same every frame)
        private val TYPE_FFFF = ",ffff\u0000\u0000\u0000"   // 4 floats, padded to 8 bytes
        private val TYPE_FFF  = ",fff\u0000"                // 3 floats, padded to 4 bytes

        // Pre-build OSC address bytes for each joint (cached at class load)
        private val JOINT_ADDRESS_BYTES: Map<Int, ByteArray> =
            MotionRecorder.HIERARCHY_ORDER.associateWith { jointIdx ->
                val name = JOINT_NAMES[jointIdx] ?: "Joint$jointIdx"
                oscString("/hand/$name/rotation")
            }
        private val WRIST_POS_ADDRESS: ByteArray = oscString("/hand/wrist/position")

        /** Pad a string to the next 4-byte boundary with null bytes, as required by OSC. */
        private fun oscString(s: String): ByteArray {
            val raw  = s.toByteArray(StandardCharsets.US_ASCII) + byteArrayOf(0)
            val pad  = (4 - raw.size % 4) % 4
            return raw + ByteArray(pad)
        }
    }

    // ─── State ────────────────────────────────────────────────────────────────

    /** True when the streamer has been started and not yet stopped. */
    var isStreaming: Boolean = false
        private set

    /** Currently configured target host. */
    var host: String = DEFAULT_HOST
        private set

    /** Currently configured target port. */
    var port: Int = DEFAULT_PORT
        private set

    private var scope:   CoroutineScope? = null
    private var socket:  DatagramSocket? = null
    private var channel: Channel<RetargetResult>? = null

    // ─── Lifecycle ────────────────────────────────────────────────────────────

    /**
     * Open a UDP socket and start the send coroutine.
     *
     * @param host Target IP address or hostname. Use `"255.255.255.255"` for LAN broadcast
     *             (requires `SO_BROADCAST` — set automatically). Use a specific IP for
     *             point-to-point (lower jitter).
     * @param port Target UDP port. Default: 9000 (standard OSC port for mocap tools).
     */
    fun start(host: String = DEFAULT_HOST, port: Int = DEFAULT_PORT) {
        if (isStreaming) stop()

        this.host = host
        this.port = port

        val ch  = Channel<RetargetResult>(Channel.CONFLATED)
        channel = ch

        val skt = DatagramSocket().apply {
            broadcast = true
            sendBufferSize = 65536
        }
        socket = skt

        val address = InetAddress.getByName(host)

        scope = CoroutineScope(Dispatchers.IO + Job()).also { sc ->
            sc.launch {
                while (isActive) {
                    val result = ch.receive()
                    try {
                        val bundle = buildBundle(result)
                        val packet = DatagramPacket(bundle, bundle.size, address, port)
                        skt.send(packet)
                    } catch (_: Exception) {
                        // Socket may have been closed — loop exits when scope is cancelled
                    }
                }
            }
        }

        isStreaming = true
    }

    /**
     * Stop streaming and close the UDP socket.
     * Safe to call multiple times or before [start].
     */
    fun stop() {
        isStreaming = false
        scope?.cancel()
        scope = null
        socket?.close()
        socket = null
        channel?.close()
        channel = null
    }

    // ─── Frame send ──────────────────────────────────────────────────────────

    /**
     * Enqueue a retarget result for transmission.
     *
     * Uses [Channel.CONFLATED]: if the send coroutine is busy, this frame replaces
     * any previously queued (unsent) frame. The tracking thread never blocks.
     *
     * Safe to call from any thread.
     */
    fun sendFrame(result: RetargetResult) {
        if (!isStreaming) return
        channel?.trySendBlocking(result)
    }

    // ─── OSC bundle builder ───────────────────────────────────────────────────

    /**
     * Build a complete OSC bundle for one [RetargetResult].
     *
     * Bundle layout:
     *   - 16-byte bundle header (#bundle + immediate timetag)
     *   - For each joint in [MotionRecorder.HIERARCHY_ORDER]:
     *       4-byte message size + OSC message (/hand/<Name>/rotation, ,ffff, x y z w)
     *   - Wrist position message:
     *       4-byte message size + OSC message (/hand/wrist/position, ,fff, x y z)
     *
     * Total size: ~16 + 17 × (address~24 + typetag~8 + data~16) = ~16 + 17×48 ≈ 832 bytes.
     * Well within a single UDP datagram on any network (MTU ≥ 1500 bytes).
     */
    private fun buildBundle(result: RetargetResult): ByteArray {
        // Pre-size the output buffer conservatively
        val buf = ByteBuffer.allocate(1200).order(ByteOrder.BIG_ENDIAN)

        buf.put(BUNDLE_HEADER)

        // Rotation messages — one per joint
        for (jointIdx in MotionRecorder.HIERARCHY_ORDER) {
            val q       = result.jointRotations[jointIdx] ?: Quaternion.IDENTITY
            val addrB   = JOINT_ADDRESS_BYTES[jointIdx]!!
            val typeB   = TYPE_FFFF.toByteArray(StandardCharsets.US_ASCII)
            val msgSize = addrB.size + typeB.size + 16   // 16 = 4 floats × 4 bytes

            buf.putInt(msgSize)
            buf.put(addrB)
            buf.put(typeB)
            buf.putFloat(q.x)
            buf.putFloat(q.y)
            buf.putFloat(q.z)
            buf.putFloat(q.w)
        }

        // Wrist position message
        val pos     = result.wristTransform.position
        val typeB   = TYPE_FFF.toByteArray(StandardCharsets.US_ASCII)
        val msgSize = WRIST_POS_ADDRESS.size + typeB.size + 12   // 12 = 3 floats × 4 bytes

        buf.putInt(msgSize)
        buf.put(WRIST_POS_ADDRESS)
        buf.put(typeB)
        buf.putFloat(pos.x)
        buf.putFloat(pos.y)
        buf.putFloat(pos.z)

        // Return only the written bytes
        val written = buf.position()
        return buf.array().copyOf(written)
    }
}
