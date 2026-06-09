package com.arhand.mocap

import android.content.Context
import android.net.Uri
import com.arhand.util.Vec3
import org.json.JSONArray
import org.json.JSONObject
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Loads a user-provided GLB asset from storage, extracts its skin joint hierarchy and
 * inverse bind matrices, and produces a [LoadedAsset] ready for [BoneRetargeter] and
 * the forthcoming [SkinnedMeshRenderer].
 *
 * Supported asset formats:
 *   - Binary glTF 2.0 (.glb) with a `skins` entry — full skeletal mesh.
 *   - GLB with no skin — mesh is loaded but retargeted via the symmetric bind pose.
 *   - No asset / null URI — falls back to [AssetLoader.DEFAULT_PUPPET].
 *
 * Thread safety: [load] is a blocking call intended to run on Dispatchers.IO.
 *
 * Usage:
 * ```kotlin
 * viewModelScope.launch(Dispatchers.IO) {
 *     val asset = AssetLoader.load(context, uri)
 *     withContext(Dispatchers.Main) { viewModel.onAssetLoaded(asset) }
 * }
 * ```
 */
object AssetLoader {

    // ─── GLB binary constants ─────────────────────────────────────────────────
    private const val GLB_MAGIC   = 0x46546C67  // "glTF"
    private const val GLB_VERSION = 2
    private const val CHUNK_JSON  = 0x4E4F534A  // "JSON"
    private const val CHUNK_BIN   = 0x004E4942  // "BIN\0"

    // ─── glTF accessor component types ───────────────────────────────────────
    private const val COMPONENT_FLOAT = 5126

    /**
     * Load a [LoadedAsset] from [uri]. Returns [DEFAULT_PUPPET] if uri is null,
     * the file is unreadable, or parsing fails.
     *
     * @param context Android context for ContentResolver access.
     * @param uri     Content URI from the file picker, or null to get the default puppet.
     */
    fun load(context: Context, uri: Uri?): LoadedAsset {
        if (uri == null) return DEFAULT_PUPPET
        return try {
            context.contentResolver.openInputStream(uri)?.use { stream ->
                parseGlb(stream)
            } ?: DEFAULT_PUPPET
        } catch (e: Exception) {
            DEFAULT_PUPPET
        }
    }

    // ─── GLB Parser ──────────────────────────────────────────────────────────

    private fun parseGlb(stream: InputStream): LoadedAsset {
        val bytes = stream.readBytes()
        val buf   = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // Validate header
        val magic   = buf.int
        val version = buf.int
        val _total  = buf.int   // total file length (unused beyond validation)
        if (magic != GLB_MAGIC || version != GLB_VERSION) return DEFAULT_PUPPET

        // Read chunks
        var jsonBytes: ByteArray? = null
        var binBytes:  ByteArray? = null

        while (buf.remaining() >= 8) {
            val chunkLen  = buf.int
            val chunkType = buf.int
            if (chunkLen < 0 || chunkLen > buf.remaining()) break
            val chunkData = ByteArray(chunkLen).also { buf.get(it) }
            when (chunkType) {
                CHUNK_JSON -> jsonBytes = chunkData
                CHUNK_BIN  -> binBytes  = chunkData
            }
        }

        val jsonRaw = jsonBytes ?: return DEFAULT_PUPPET
        val root    = JSONObject(String(jsonRaw, Charsets.UTF_8))

        return buildAsset(root, binBytes)
    }

    // ─── Asset builder ────────────────────────────────────────────────────────

    private fun buildAsset(root: JSONObject, bin: ByteArray?): LoadedAsset {
        val bufferViews = root.optJSONArray("bufferViews")
        val accessors   = root.optJSONArray("accessors")

        // ── Mesh geometry ─────────────────────────────────────────────────────
        val meshPositions = extractMeshPositions(root, accessors, bufferViews, bin)

        // ── Skin + bind pose ──────────────────────────────────────────────────
        val skins = root.optJSONArray("skins")
        if (skins == null || skins.length() == 0) {
            // No skeleton — use mesh with symmetric bind pose
            return LoadedAsset(
                meshPositions = meshPositions,
                jointNodeIndices = emptyList(),
                inverseBindMatrices = emptyList(),
                bindPose = BoneRetargeter.symmetricBindPose(),
                hasSkin = false
            )
        }

        val skin        = skins.getJSONObject(0)
        val joints      = skin.getJSONArray("joints")
        val ibmAccessor = skin.optInt("inverseBindMatrices", -1)

        val jointNodeIndices = (0 until joints.length()).map { joints.getInt(it) }

        // Extract inverse bind matrices (one 4×4 float matrix per joint)
        val ibmList: List<FloatArray> = if (ibmAccessor >= 0 && accessors != null && bin != null) {
            extractMat4Array(accessors, bufferViews, bin, ibmAccessor, jointNodeIndices.size)
        } else {
            List(jointNodeIndices.size) { identityMatrix() }
        }

        // Derive bind pose bone directions from the inverse bind matrices
        val bindPose = deriveBindPose(root, jointNodeIndices, ibmList)

        return LoadedAsset(
            meshPositions       = meshPositions,
            jointNodeIndices    = jointNodeIndices,
            inverseBindMatrices = ibmList,
            bindPose            = bindPose,
            hasSkin             = true
        )
    }

    // ─── Geometry extraction ─────────────────────────────────────────────────

    private fun extractMeshPositions(
        root: JSONObject,
        accessors: JSONArray?,
        bufferViews: JSONArray?,
        bin: ByteArray?
    ): FloatArray {
        if (accessors == null || bufferViews == null || bin == null) return FloatArray(0)

        val meshes = root.optJSONArray("meshes") ?: return FloatArray(0)
        if (meshes.length() == 0) return FloatArray(0)

        val prim       = meshes.getJSONObject(0)
            .optJSONArray("primitives")?.getJSONObject(0) ?: return FloatArray(0)
        val posAccIdx  = prim.optJSONObject("attributes")?.optInt("POSITION", -1) ?: -1
        if (posAccIdx < 0) return FloatArray(0)

        return extractFloatVec3(accessors, bufferViews, bin, posAccIdx)
    }

    private fun extractFloatVec3(
        accessors: JSONArray,
        bufferViews: JSONArray,
        bin: ByteArray,
        accessorIdx: Int
    ): FloatArray {
        if (accessorIdx >= accessors.length()) return FloatArray(0)
        val acc      = accessors.getJSONObject(accessorIdx)
        if (acc.optInt("componentType") != COMPONENT_FLOAT) return FloatArray(0)
        if (acc.optString("type") != "VEC3") return FloatArray(0)

        val count    = acc.getInt("count")
        val bvIdx    = acc.getInt("bufferView")
        val byteOff  = acc.optInt("byteOffset", 0)

        if (bvIdx >= bufferViews.length()) return FloatArray(0)
        val bv       = bufferViews.getJSONObject(bvIdx)
        val bvOffset = bv.optInt("byteOffset", 0)
        val stride   = bv.optInt("byteStride", 12)  // 12 = 3 floats × 4 bytes

        val result = FloatArray(count * 3)
        val bbuf   = ByteBuffer.wrap(bin, bvOffset + byteOff, bin.size - bvOffset - byteOff)
            .order(ByteOrder.LITTLE_ENDIAN)

        for (i in 0 until count) {
            result[i * 3]     = bbuf.getFloat(bvOffset + byteOff + i * stride)
            result[i * 3 + 1] = bbuf.getFloat(bvOffset + byteOff + i * stride + 4)
            result[i * 3 + 2] = bbuf.getFloat(bvOffset + byteOff + i * stride + 8)
        }
        return result
    }

    private fun extractMat4Array(
        accessors: JSONArray,
        bufferViews: JSONArray?,
        bin: ByteArray,
        accessorIdx: Int,
        count: Int
    ): List<FloatArray> {
        if (bufferViews == null || accessorIdx >= accessors.length())
            return List(count) { identityMatrix() }

        val acc     = accessors.getJSONObject(accessorIdx)
        val bvIdx   = acc.optInt("bufferView", -1)
        val byteOff = acc.optInt("byteOffset", 0)
        if (bvIdx < 0 || bvIdx >= bufferViews.length()) return List(count) { identityMatrix() }

        val bv      = bufferViews.getJSONObject(bvIdx)
        val bvOff   = bv.optInt("byteOffset", 0)
        val base    = bvOff + byteOff

        return List(count) { i ->
            val mat = FloatArray(16)
            val bbuf = ByteBuffer.wrap(bin, base + i * 64, 64).order(ByteOrder.LITTLE_ENDIAN)
            for (j in 0 until 16) mat[j] = bbuf.float
            mat
        }
    }

    // ─── Bind pose derivation ─────────────────────────────────────────────────

    /**
     * Derive [BindPose] bone directions from a glTF node hierarchy.
     *
     * Strategy: for each [BoneRetargeter] JOINT_* bone segment (base→tip), look up the
     * asset's joint node index for both ends of that segment, read their translations
     * from the glTF nodes array, and compute the unit direction vector.
     *
     * If the node indices can't be mapped (asset uses a different naming convention),
     * falls back to the symmetric bind pose for those joints.
     */
    private fun deriveBindPose(
        root: JSONObject,
        jointNodeIndices: List<Int>,
        ibmList: List<FloatArray>
    ): BindPose {
        val nodes = root.optJSONArray("nodes")

        // Map each joint node index to its translation (bind-pose position)
        // from its inverse bind matrix (last column = -R^T * t).
        val jointPositions: Map<Int, Vec3> = jointNodeIndices.mapIndexed { slotIdx, nodeIdx ->
            val ibm = ibmList.getOrNull(slotIdx) ?: identityMatrix()
            // Extract translation from IBM: t = -R * lastColumn(IBM)
            // For column-major mat4, last column is indices 12,13,14
            val pos = Vec3(ibm[12], ibm[13], ibm[14])
            nodeIdx to pos
        }.toMap()

        // Try to match glTF joint node indices to BoneRetargeter bone segments.
        // Heuristic: use the joint index order — most hand rigs follow MediaPipe
        // or VRChat order (wrist first, then thumb CMC→MCP→IP, index MCP→PIP→DIP, …).
        // For assets that don't match, the symmetric fallback fills in the gaps.
        val fallback = BoneRetargeter.symmetricBindPose()
        val dirs = mutableMapOf<Int, Vec3>()

        // The first `min(JOINT_COUNT, jointNodeIndices.size)` joint slots are mapped
        // in BoneRetargeter order. Each bone direction = normalize(tip_pos - base_pos).
        for ((jointIdx, baseIdx, tipIdx) in BoneRetargeter.BONE_SEGMENTS) {
            val basePos = jointPositions[jointNodeIndices.getOrElse(baseIdx) { -1 }]
            val tipPos  = jointPositions[jointNodeIndices.getOrElse(tipIdx)  { -1 }]

            dirs[jointIdx] = if (basePos != null && tipPos != null) {
                val d = (tipPos - basePos)
                if (d.length() > 1e-6f) d.normalized()
                else fallback.boneDirections[jointIdx] ?: Vec3(0f, 1f, 0f)
            } else {
                fallback.boneDirections[jointIdx] ?: Vec3(0f, 1f, 0f)
            }
        }

        val wristPos = jointPositions[jointNodeIndices.firstOrNull() ?: -1]
            ?: fallback.wristPosition

        return BindPose(dirs, wristPos)
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private fun identityMatrix(): FloatArray = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 1f, 0f,
        0f, 0f, 0f, 1f
    )

    // ─── Default puppet ───────────────────────────────────────────────────────

    /**
     * Built-in fallback asset used when no user GLB is loaded.
     *
     * The puppet has no mesh geometry (empty positions) — it drives retargeting only,
     * so the live skeleton lines from [HandRenderer] serve as the visual stand-in until
     * a real asset is loaded. [SkinnedMeshRenderer] should show nothing (or the skeleton)
     * when [LoadedAsset.meshPositions] is empty.
     */
    val DEFAULT_PUPPET = LoadedAsset(
        meshPositions       = FloatArray(0),
        jointNodeIndices    = emptyList(),
        inverseBindMatrices = emptyList(),
        bindPose            = BoneRetargeter.symmetricBindPose(),
        hasSkin             = false
    )
}

// ─── LoadedAsset ─────────────────────────────────────────────────────────────

/**
 * The fully parsed result of [AssetLoader.load].
 *
 * Passed to [BoneRetargeter] (for the bind pose) and to [SkinnedMeshRenderer]
 * (for mesh geometry + inverse bind matrices).
 */
data class LoadedAsset(
    /**
     * Flat float array of mesh vertex positions (x, y, z × vertexCount).
     * Empty for the default puppet.
     */
    val meshPositions: FloatArray,

    /**
     * Ordered list of glTF node indices that make up the skin's joint hierarchy.
     * Index 0 = wrist root; order matches BoneRetargeter JOINT_* constants when
     * the asset uses standard hand rig layout.
     */
    val jointNodeIndices: List<Int>,

    /**
     * Inverse bind matrices — one column-major 4×4 float matrix per joint.
     * Used by the GPU skinning shader to transform vertices from model space
     * to joint space before applying the live rotation.
     */
    val inverseBindMatrices: List<FloatArray>,

    /**
     * Bind pose derived from the skin's joint hierarchy, ready for [BoneRetargeter].
     */
    val bindPose: BindPose,

    /** False when asset has no skin — retargeting still works via symmetric bind pose. */
    val hasSkin: Boolean
) {
    /** Number of skin joints. */
    val jointCount: Int get() = jointNodeIndices.size

    /** Vertex count derived from flat positions array. */
    val vertexCount: Int get() = meshPositions.size / 3

    override fun equals(other: Any?) = other is LoadedAsset &&
        meshPositions.contentEquals(other.meshPositions) &&
        jointNodeIndices == other.jointNodeIndices &&
        hasSkin == other.hasSkin

    override fun hashCode(): Int {
        var result = meshPositions.contentHashCode()
        result = 31 * result + jointNodeIndices.hashCode()
        result = 31 * result + hasSkin.hashCode()
        return result
    }
}
