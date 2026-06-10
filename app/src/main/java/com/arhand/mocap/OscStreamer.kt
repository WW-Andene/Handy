package com.arhand.mocap

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.channels.Channel
// trySendBlocking removed — trySend() used instead (never throws on closed/conflated channel)
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

// ─── OSC operating mode ───────────────────────────────────────────────────────

/**
 * G3 — Whether this device is sending or receiving hand pose data over OSC.
 *
 * | Mode    | Behaviour                                                        |
 * |---------|------------------------------------------------------------------|
 * | SEND    | Existing behaviour — broadcast joint rotations to a remote host  |
 * | RECEIVE | Listen on UDP port; render remote hand pose instead of camera    |
 */
enum class OscMode {
    SEND,
    RECEIVE;

    val label: String get() = when (this) {
        SEND    -> "SND"
        RECEIVE -> "RCV"
    }

    val description: String get() = when (this) {
        SEND    -> "Broadcast (send to remote)"
        RECEIVE -> "Listen (render remote hand)"
    }
}

// ─── Schema enum ─────────────────────────────────────────────────────────────

/**
 * OSC address schema presets.
 *
 * | Schema            | Compatible receiver                                         |
 * |-------------------|-------------------------------------------------------------|
 * | HANDY_DEFAULT     | Any OSC client; Handy's native `/hand/<Joint>/rotation`     |
 * | VMC               | VirtualMotionCapture, VSeeFace, 3tene, Warudo — VMC v2.3    |
 * | UNREAL_LIVE_LINK  | Unreal Engine 5 LiveLink OSC Source plugin                  |
 * | VSEEFACE          | VSeeFace legacy `/VMC/Ext/Bone/Pos` direct channel          |
 *
 * VMC is the recommended default for VTubing. Unreal LiveLink is recommended for
 * game-engine character control. VSeeFace legacy is kept for compatibility with
 * older setups already configured on a fixed address schema.
 */
enum class OscSchema {
    HANDY_DEFAULT,
    VMC,
    UNREAL_LIVE_LINK,
    VSEEFACE;

    /** Short display label for UI chips. */
    val label: String get() = when (this) {
        HANDY_DEFAULT    -> "HANDY"
        VMC              -> "VMC"
        UNREAL_LIVE_LINK -> "UE LL"
        VSEEFACE         -> "VSF"
    }

    /** Longer description for settings/tooltips. */
    val description: String get() = when (this) {
        HANDY_DEFAULT    -> "Handy native (/hand/<Joint>/rotation)"
        VMC              -> "VirtualMotionCapture / VSeeFace v2.3 (/VMC/Ext/Bon/Rot)"
        UNREAL_LIVE_LINK -> "Unreal LiveLink OSC (/LiveLink/<Subject>/Bone/<Joint>)"
        VSEEFACE         -> "VSeeFace legacy (/VMC/Ext/Bone/Pos)"
    }
}

// ─── Schema address tables ────────────────────────────────────────────────────

/**
 * Maps schema → (jointIndex → OSC address bytes).
 * Built once at class load; all entries are pre-padded to 4-byte boundaries.
 */
private object SchemaAddresses {

    // --- Handy native ---
    // /hand/<JointName>/rotation
    private val handyJointNames = mapOf(
        BoneRetargeter.JOINT_WRIST       to "Wrist",
        BoneRetargeter.JOINT_THUMB_CMC   to "ThumbCMC",
        BoneRetargeter.JOINT_THUMB_MCP   to "ThumbMCP",
        BoneRetargeter.JOINT_THUMB_IP    to "ThumbIP",
        BoneRetargeter.JOINT_INDEX_MCP   to "IndexMCP",
        BoneRetargeter.JOINT_INDEX_PIP   to "IndexPIP",
        BoneRetargeter.JOINT_INDEX_DIP   to "IndexDIP",
        BoneRetargeter.JOINT_MIDDLE_MCP  to "MiddleMCP",
        BoneRetargeter.JOINT_MIDDLE_PIP  to "MiddlePIP",
        BoneRetargeter.JOINT_MIDDLE_DIP  to "MiddleDIP",
        BoneRetargeter.JOINT_RING_MCP    to "RingMCP",
        BoneRetargeter.JOINT_RING_PIP    to "RingPIP",
        BoneRetargeter.JOINT_RING_DIP    to "RingDIP",
        BoneRetargeter.JOINT_PINKY_MCP   to "PinkyMCP",
        BoneRetargeter.JOINT_PINKY_PIP   to "PinkyPIP",
        BoneRetargeter.JOINT_PINKY_DIP   to "PinkyDIP"
    )

    // --- VMC v2.3 bone names ---
    // Reference: https://protocol.vmc.info/marionette-spec  (VirtualMotionCapture OSC Protocol)
    // Address: /VMC/Ext/Bon/Rot   args: string boneName, float qx, qy, qz, qw
    // The bone name is sent as an OSC string argument, not encoded in the address.
    // We still need per-joint address bytes (all the same "/VMC/Ext/Bon/Rot") and
    // a pre-built bone-name string argument per joint.
    private val vmcBoneNames = mapOf(
        BoneRetargeter.JOINT_WRIST       to "RightHand",
        BoneRetargeter.JOINT_THUMB_CMC   to "RightThumbProximal",
        BoneRetargeter.JOINT_THUMB_MCP   to "RightThumbIntermediate",
        BoneRetargeter.JOINT_THUMB_IP    to "RightThumbDistal",
        BoneRetargeter.JOINT_INDEX_MCP   to "RightIndexProximal",
        BoneRetargeter.JOINT_INDEX_PIP   to "RightIndexIntermediate",
        BoneRetargeter.JOINT_INDEX_DIP   to "RightIndexDistal",
        BoneRetargeter.JOINT_MIDDLE_MCP  to "RightMiddleProximal",
        BoneRetargeter.JOINT_MIDDLE_PIP  to "RightMiddleIntermediate",
        BoneRetargeter.JOINT_MIDDLE_DIP  to "RightMiddleDistal",
        BoneRetargeter.JOINT_RING_MCP    to "RightRingProximal",
        BoneRetargeter.JOINT_RING_PIP    to "RightRingIntermediate",
        BoneRetargeter.JOINT_RING_DIP    to "RightRingDistal",
        BoneRetargeter.JOINT_PINKY_MCP   to "RightLittleProximal",
        BoneRetargeter.JOINT_PINKY_PIP   to "RightLittleIntermediate",
        BoneRetargeter.JOINT_PINKY_DIP   to "RightLittleDistal"
    )

    // --- Unreal LiveLink OSC bone names ---
    // Address: /LiveLink/<SubjectName>/Bone/<BoneName>  args: float qx, qy, qz, qw
    // Subject name is configurable; we default to "Hand".
    // UE LiveLink uses Mixamo-compatible bone names for hand rigs.
    private val unrealBoneNames = mapOf(
        BoneRetargeter.JOINT_WRIST       to "RightHand",
        BoneRetargeter.JOINT_THUMB_CMC   to "RightHandThumb1",
        BoneRetargeter.JOINT_THUMB_MCP   to "RightHandThumb2",
        BoneRetargeter.JOINT_THUMB_IP    to "RightHandThumb3",
        BoneRetargeter.JOINT_INDEX_MCP   to "RightHandIndex1",
        BoneRetargeter.JOINT_INDEX_PIP   to "RightHandIndex2",
        BoneRetargeter.JOINT_INDEX_DIP   to "RightHandIndex3",
        BoneRetargeter.JOINT_MIDDLE_MCP  to "RightHandMiddle1",
        BoneRetargeter.JOINT_MIDDLE_PIP  to "RightHandMiddle2",
        BoneRetargeter.JOINT_MIDDLE_DIP  to "RightHandMiddle3",
        BoneRetargeter.JOINT_RING_MCP    to "RightHandRing1",
        BoneRetargeter.JOINT_RING_PIP    to "RightHandRing2",
        BoneRetargeter.JOINT_RING_DIP    to "RightHandRing3",
        BoneRetargeter.JOINT_PINKY_MCP   to "RightHandPinky1",
        BoneRetargeter.JOINT_PINKY_PIP   to "RightHandPinky2",
        BoneRetargeter.JOINT_PINKY_DIP   to "RightHandPinky3"
    )

    // --- VSeeFace legacy: same addresses as VMC, kept as a separate preset so
    //     users on the VSeeFace "Direct" OSC tab (which does not follow v2.3 spec
    //     fully) can select it without touching VMC settings. Address is
    //     /VMC/Ext/Bone/Pos (note: "Bone" not "Bon", and "Pos" not "Rot") —
    //     VSeeFace legacy sends position + quaternion as a single blob.
    // We encode it the same as VMC but with the legacy address so the receiver
    // picks it up on the correct address filter.
    // Reference: VSeeFace OSC documentation (https://www.vseeface.icu/#osc-control)
    // ─────────────────────────────────────────────────────────────────────────

    // Pre-built address byte arrays ───────────────────────────────────────────

    val handyRotAddr:  Map<Int, ByteArray> = MotionRecorder.HIERARCHY_ORDER.associateWith { j ->
        val name = handyJointNames[j] ?: "Joint$j"
        oscString("/hand/$name/rotation")
    }
    val handyPosAddr:  ByteArray = oscString("/hand/wrist/position")

    // VMC: single address "/VMC/Ext/Bon/Rot" for all joints; bone name in args
    val vmcRotAddr:    ByteArray = oscString("/VMC/Ext/Bon/Rot")
    val vmcPosAddr:    ByteArray = oscString("/VMC/Ext/Bon/Rot")   // wrist pos also via Bon/Rot
    val vmcBoneBytes:  Map<Int, ByteArray> = MotionRecorder.HIERARCHY_ORDER.associateWith { j ->
        oscString(vmcBoneNames[j] ?: "RightHand")
    }
    val vmcWristBone:  ByteArray = oscString("RightHand")

    // Unreal LiveLink: address per bone "/LiveLink/Hand/Bone/<BoneName>"
    val unrealRotAddr: Map<Int, ByteArray> = MotionRecorder.HIERARCHY_ORDER.associateWith { j ->
        val bone = unrealBoneNames[j] ?: "RightHand"
        oscString("/LiveLink/Hand/Bone/$bone")
    }
    val unrealPosAddr: ByteArray = oscString("/LiveLink/Hand/Root")

    // VSeeFace legacy: "/VMC/Ext/Bone/Pos" with bone name in args
    val vsfRotAddr:    ByteArray = oscString("/VMC/Ext/Bone/Pos")
    val vsfPosAddr:    ByteArray = oscString("/VMC/Ext/Bone/Pos")
    // VSeeFace reuses VMC bone names
    val vsfBoneBytes:  Map<Int, ByteArray> = vmcBoneBytes
    val vsfWristBone:  ByteArray = vmcWristBone

    /** Pad a string to the next 4-byte boundary with null bytes, as required by OSC 1.0. */
    fun oscString(s: String): ByteArray {
        val raw = s.toByteArray(StandardCharsets.US_ASCII) + byteArrayOf(0)
        val pad = (4 - raw.size % 4) % 4
        return raw + ByteArray(pad)
    }
}

// ─── OscStreamer ──────────────────────────────────────────────────────────────

/**
 * Streams live hand joint rotations as OSC (Open Sound Control) messages over UDP.
 *
 * OSC is the standard protocol used by professional mocap systems (Vicon, OptiTrack)
 * and is natively supported by Unreal Engine, Unity (via OSCJack / extOSC), TouchDesigner,
 * and Notch. This allows Handy to drive a desktop character rig in real time without a
 * USB connection or file export step.
 *
 * ## Schema presets
 *
 * The [schema] property selects the OSC address layout used when building bundles:
 *
 * | [OscSchema]        | Address pattern                              | Use case               |
 * |--------------------|----------------------------------------------|------------------------|
 * | HANDY_DEFAULT      | `/hand/<Joint>/rotation`                     | Custom receivers       |
 * | VMC                | `/VMC/Ext/Bon/Rot` + bone name arg           | VTubing (VSeeFace etc) |
 * | UNREAL_LIVE_LINK   | `/LiveLink/Hand/Bone/<BoneName>`             | Unreal Engine 5        |
 * | VSEEFACE           | `/VMC/Ext/Bone/Pos` + bone name arg          | VSeeFace legacy        |
 *
 * [schema] can be changed at any time, even while streaming — the next [sendFrame]
 * call will use the new schema.
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
 * streamer.schema = OscSchema.VMC
 * streamer.start(host = "192.168.1.42", port = 39539)  // VMC default port
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
 * [schema] is a plain var — write it from the main thread before or between frames.
 */
class OscStreamer {

    companion object {
        const val DEFAULT_PORT       = 9000
        const val DEFAULT_HOST       = "255.255.255.255"   // LAN broadcast

        /** VMC protocol default port (VirtualMotionCapture, VSeeFace). */
        const val VMC_PORT           = 39539
        /** Unreal LiveLink OSC default port. */
        const val UNREAL_LL_PORT     = 7771

        /** Suggested default port for each schema. */
        fun defaultPortFor(schema: OscSchema): Int = when (schema) {
            OscSchema.VMC              -> VMC_PORT
            OscSchema.VSEEFACE         -> VMC_PORT
            OscSchema.UNREAL_LIVE_LINK -> UNREAL_LL_PORT
            OscSchema.HANDY_DEFAULT    -> DEFAULT_PORT
        }

        // OSC bundle header: "#bundle\0" + timetag (immediate = 1)
        private val BUNDLE_HEADER: ByteArray = run {
            val buf = ByteBuffer.allocate(16).order(ByteOrder.BIG_ENDIAN)
            buf.put("#bundle\u0000".toByteArray(StandardCharsets.US_ASCII))
            buf.putInt(0)   // timetag seconds (immediate)
            buf.putInt(1)   // timetag fraction (immediate sentinel)
            buf.array()
        }

        // Pre-build type tag strings (immutable — same every frame)
        // ,ffff padded to 8 bytes  (4 floats = quaternion)
        private val TYPE_FFFF = ",ffff\u0000\u0000\u0000".toByteArray(StandardCharsets.US_ASCII)
        // ,fff  padded to 8 bytes  (3 floats = position)
        // OSC 1.0 §3.1.2: type tag strings are null-terminated then padded to a 4-byte boundary.
        // ",fff" = 4 chars + 1 null = 5 bytes → next 4-byte boundary = 8 bytes (3 pad nulls).
        // The previous value had only 1 null (5 bytes total), which caused every OSC receiver
        // (TouchDesigner, Python-osc, Unreal LiveLink) to misparse wrist position floats.
        private val TYPE_FFF  = ",fff\u0000\u0000\u0000\u0000".toByteArray(StandardCharsets.US_ASCII)
        // ,sffff padded to 8 bytes (1 string arg + 4 floats = VMC bone rotation)
        // The string arg is written inline; type tag length stays constant only
        // because we pre-size the string arg to a fixed-width oscString.
        private val TYPE_SFFFF = ",sffff\u0000\u0000".toByteArray(StandardCharsets.US_ASCII)
        // ,sfff padded to 8 bytes (1 string arg + 3 floats = VMC wrist pos)
        private val TYPE_SFFF  = ",sfff\u0000\u0000\u0000".toByteArray(StandardCharsets.US_ASCII)
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

    /**
     * Active OSC address schema.
     *
     * Safe to change at any time — even during active streaming. The next
     * [sendFrame] call picks up the new value. No restart required.
     */
    @Volatile
    var schema: OscSchema = OscSchema.HANDY_DEFAULT

    private var scope:   CoroutineScope? = null
    private var socket:  DatagramSocket? = null
    private var channel: Channel<RetargetResult>? = null

    // ─── Lifecycle ────────────────────────────────────────────────────────────

    /**
     * Open a UDP socket and start the send coroutine.
     *
     * @param host   Target IP address or hostname. Use `"255.255.255.255"` for LAN
     *               broadcast (requires `SO_BROADCAST` — set automatically). Use a
     *               specific IP for point-to-point (lower jitter).
     * @param port   Target UDP port. Defaults to [defaultPortFor] the current [schema].
     * @param schema Optional schema override — sets [schema] before starting.
     */
    fun start(
        host:   String    = DEFAULT_HOST,
        schema: OscSchema = this.schema,
        port:   Int       = defaultPortFor(schema)
    ) {
        if (isStreaming) stop()

        this.host   = host
        this.port   = port
        this.schema = schema

        val ch  = Channel<RetargetResult>(Channel.CONFLATED)
        channel = ch

        val skt = DatagramSocket().apply {
            broadcast    = true
            sendBufferSize = 65536
        }
        socket = skt

        val address = try {
            InetAddress.getByName(host)
        } catch (e: Exception) {
            skt.close()
            socket  = null
            channel?.close()
            channel = null
            return
        }

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
        scope   = null
        socket?.close()
        socket  = null
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
        channel?.trySend(result)
    }

    // ─── OSC bundle builder ───────────────────────────────────────────────────

    /**
     * Dispatch bundle construction to the active [schema].
     */
    private fun buildBundle(result: RetargetResult): ByteArray = when (schema) {
        OscSchema.HANDY_DEFAULT    -> buildBundleHandy(result)
        OscSchema.VMC              -> buildBundleVmc(result, SchemaAddresses.vmcRotAddr,
                                                              SchemaAddresses.vmcBoneBytes,
                                                              SchemaAddresses.vmcWristBone,
                                                              SchemaAddresses.vmcPosAddr)
        OscSchema.VSEEFACE         -> buildBundleVmc(result, SchemaAddresses.vsfRotAddr,
                                                              SchemaAddresses.vsfBoneBytes,
                                                              SchemaAddresses.vsfWristBone,
                                                              SchemaAddresses.vsfPosAddr)
        OscSchema.UNREAL_LIVE_LINK -> buildBundleUnreal(result)
    }

    // ── HANDY_DEFAULT ─────────────────────────────────────────────────────────
    //
    // One message per joint:
    //   address  "/hand/<JointName>/rotation"   (pre-built oscString)
    //   typetag  ",ffff"
    //   data     qx qy qz qw  (4 × float32 BE)
    // + wrist position message:
    //   address  "/hand/wrist/position"
    //   typetag  ",fff"
    //   data     x y z  (3 × float32 BE)
    //
    // Total: ~16 + 17 × (addr~24 + typetag~8 + data~16) ≈ 832 bytes

    private fun buildBundleHandy(result: RetargetResult): ByteArray {
        val buf = ByteBuffer.allocate(1200).order(ByteOrder.BIG_ENDIAN)
        buf.put(BUNDLE_HEADER)

        for (j in MotionRecorder.HIERARCHY_ORDER) {
            val q       = result.jointRotations[j] ?: Quaternion.IDENTITY
            val addrB   = SchemaAddresses.handyRotAddr[j]!!
            val msgSize = addrB.size + TYPE_FFFF.size + 16
            buf.putInt(msgSize)
            buf.put(addrB)
            buf.put(TYPE_FFFF)
            buf.putFloat(q.x); buf.putFloat(q.y); buf.putFloat(q.z); buf.putFloat(q.w)
        }

        val pos     = result.wristTransform.position
        val msgSize = SchemaAddresses.handyPosAddr.size + TYPE_FFF.size + 12
        buf.putInt(msgSize)
        buf.put(SchemaAddresses.handyPosAddr)
        buf.put(TYPE_FFF)
        buf.putFloat(pos.x); buf.putFloat(pos.y); buf.putFloat(pos.z)

        return buf.array().copyOf(buf.position())
    }

    // ── VMC / VSEEFACE ────────────────────────────────────────────────────────
    //
    // VMC v2.3 spec: /VMC/Ext/Bon/Rot  s:boneName f:qx f:qy f:qz f:qw
    // VSeeFace legacy: /VMC/Ext/Bone/Pos  s:boneName f:qx f:qy f:qz f:qw
    //
    // Both share the same message structure; only the address bytes differ.
    // Args: [oscString boneName] [float qx] [float qy] [float qz] [float qw]
    //
    // Total: ~16 + 16 × (addr~20 + typetag~8 + boneName~28 + data~16) ≈ 1160 bytes

    private fun buildBundleVmc(
        result:    RetargetResult,
        rotAddr:   ByteArray,
        boneBytes: Map<Int, ByteArray>,
        wristBone: ByteArray,
        posAddr:   ByteArray
    ): ByteArray {
        val buf = ByteBuffer.allocate(1600).order(ByteOrder.BIG_ENDIAN)
        buf.put(BUNDLE_HEADER)

        for (j in MotionRecorder.HIERARCHY_ORDER) {
            val q       = result.jointRotations[j] ?: Quaternion.IDENTITY
            val boneB   = boneBytes[j]!!
            val msgSize = rotAddr.size + TYPE_SFFFF.size + boneB.size + 16
            buf.putInt(msgSize)
            buf.put(rotAddr)
            buf.put(TYPE_SFFFF)
            buf.put(boneB)
            buf.putFloat(q.x); buf.putFloat(q.y); buf.putFloat(q.z); buf.putFloat(q.w)
        }

        // Wrist root position — same pattern, bone = "RightHand"
        val pos     = result.wristTransform.position
        val msgSize = posAddr.size + TYPE_SFFF.size + wristBone.size + 12
        buf.putInt(msgSize)
        buf.put(posAddr)
        buf.put(TYPE_SFFF)
        buf.put(wristBone)
        buf.putFloat(pos.x); buf.putFloat(pos.y); buf.putFloat(pos.z)

        return buf.array().copyOf(buf.position())
    }

    // ── UNREAL_LIVE_LINK ──────────────────────────────────────────────────────
    //
    // UE5 LiveLink OSC plugin: /LiveLink/<Subject>/Bone/<BoneName>  f:qx f:qy f:qz f:qw
    // Subject defaults to "Hand"; configurable via unrealSubject field if needed.
    // Address is pre-built per joint; no string argument needed.
    //
    // Total: ~16 + 16 × (addr~40 + typetag~8 + data~16) ≈ 1040 bytes

    private fun buildBundleUnreal(result: RetargetResult): ByteArray {
        val buf = ByteBuffer.allocate(1400).order(ByteOrder.BIG_ENDIAN)
        buf.put(BUNDLE_HEADER)

        for (j in MotionRecorder.HIERARCHY_ORDER) {
            val q       = result.jointRotations[j] ?: Quaternion.IDENTITY
            val addrB   = SchemaAddresses.unrealRotAddr[j]!!
            val msgSize = addrB.size + TYPE_FFFF.size + 16
            buf.putInt(msgSize)
            buf.put(addrB)
            buf.put(TYPE_FFFF)
            buf.putFloat(q.x); buf.putFloat(q.y); buf.putFloat(q.z); buf.putFloat(q.w)
        }

        // Root position on dedicated address
        val pos     = result.wristTransform.position
        val msgSize = SchemaAddresses.unrealPosAddr.size + TYPE_FFF.size + 12
        buf.putInt(msgSize)
        buf.put(SchemaAddresses.unrealPosAddr)
        buf.put(TYPE_FFF)
        buf.putFloat(pos.x); buf.putFloat(pos.y); buf.putFloat(pos.z)

        return buf.array().copyOf(buf.position())
    }
}
