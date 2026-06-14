package com.arhand.feature.scan

import com.arhand.scanner.HandBiometrics
import com.arhand.scanner.JointRomData

/**
 * Immutable state snapshot for the scan feature.
 *
 * Collected independently by [com.arhand.ui.DepthScannerPanel] and
 * [com.arhand.ui.ScanResultModal] — changes here don't recompose unrelated screens.
 */
data class ScanState(
    val isActive:              Boolean                  = false,
    val hasStoredModel:        Boolean                  = false,
    val exportedGlbPath:       String?                  = null,
    val handBiometrics:        HandBiometrics?          = null,
    val biometricHistory:      List<com.arhand.scanner.BiometricHistoryEntry> = emptyList(),
    val jointRomData:          JointRomData?            = null,
    val neuralReconEnabled:    Boolean                  = false,
    val neuralReconDiagnostics: NeuralReconDiagnostics? = null,
    val photoStereoEnabled:    Boolean                  = false,
    val depthMode:             Boolean                  = false,
    val depthApiAvailable:     Boolean                  = false,
    val arcoreDepthHw:         Boolean                  = false,
    /** LIMIT-2 — True while a freeform (continuous) scan is accumulating frames. */
    val freeformActive:        Boolean                  = false,
    /** LIMIT-2 — Latest status from [com.arhand.scanner.FreeformScanner]. */
    val freeformStatus:        com.arhand.scanner.FreeformScanner.FreeformStatus? = null
)
