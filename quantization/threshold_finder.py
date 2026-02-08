import numpy as np
from matplotlib import pyplot as plt
from numba import njit, prange
import unittest
from visualization import dot_figure

@njit(parallel=True)
def compute_QBin_vector(reference_distribution_P, RBinScale, maxIntBin):
    """
    Compute QBin_vector.

    Args:
    - reference_distribution_P: Reference distribution at the current iBin (1D array).
    - RBinScale: Corresponding RBinScale (1D integer array).
    - maxIntBin: Maximum integer bin value.

    Returns:
    - QBin_vector: Computed QBin_vector (1D array).
    """
    QBin_vector = np.zeros(maxIntBin)
    for iBinScale2 in prange(1, maxIntBin + 1):
        total = 0.0
        count = 0
        for j in range(RBinScale.size):
            if RBinScale[j] == iBinScale2 and reference_distribution_P[j] != 0:
                total += reference_distribution_P[j]
                count += 1
        if count > 0:
            QBin_vector[iBinScale2 - 1] = total / count
        else:
            QBin_vector[iBinScale2 - 1] = 0.0
    return QBin_vector

class FindThreshold:
    # Class-level dictionary for storing data for different layers
    KL_Layers = {}
    QAct_params = {}

    @staticmethod
    def findOptimalThreshold(features_dict, iLayer, Qmax, nRBin, startSearchPoint, ifold):
        # Extract and preprocess feature maps
        curFeaMapFlatten = features_dict[f'conv{iLayer}'].flatten()
        curFeaMapFlatten = curFeaMapFlatten[curFeaMapFlatten != 0]
        curDistributionN_P, curDistributionEdge_P = np.histogram(np.abs(curFeaMapFlatten), bins=nRBin)


        maxIntBin = Qmax + 1
        KLEntropy = np.full((nRBin + 1, 1), np.nan)
        QScaleVector = np.full((nRBin + 1, 1), np.nan)

        for iBin in range(startSearchPoint, nRBin + 1):
            # Construct reference_distribution_P and handle outliers
            reference_distribution_P = curDistributionN_P[:iBin].copy()
            if iBin + 1 < len(curDistributionN_P):
                outlier_count = np.sum(curDistributionN_P[iBin + 1:])
                reference_distribution_P[-1] += outlier_count

            # Normalize reference_distribution_P
            sum_P = np.sum(reference_distribution_P)
            if sum_P > 0.0:
                reference_distribution_P_norm = reference_distribution_P / sum_P
            else:
                reference_distribution_P_norm = reference_distribution_P  # Avoid division by zero

            # Obtain reference_distribution_P_edge
            reference_distribution_P_edge = curDistributionEdge_P[:iBin + 1]

            # Compute QBinScale
            max_edge = np.max(reference_distribution_P_edge)
            if max_edge > 0.0:
                QBinScale = maxIntBin / max_edge
            else:
                QBinScale = 1.0  # Avoid division by zero

            # Compute RBinScale
            RBinScale = np.round(reference_distribution_P_edge * QBinScale).astype(int)
            RBinScale = RBinScale[1:]
            RBinScale[RBinScale == 0] = 1

            # Call the Numba-optimized helper function to compute QBin_vector
            QBin_vector = compute_QBin_vector(reference_distribution_P, RBinScale, maxIntBin)

            # Handle NaN values
            QBin_vector = np.nan_to_num(QBin_vector, nan=0.0)

            # Compute newQBinScale
            invQBinScale = maxIntBin / len(reference_distribution_P)
            newQBinScale = np.round(np.arange(invQBinScale, maxIntBin + invQBinScale, invQBinScale)).astype(int)
            if newQBinScale.size != reference_distribution_P_norm.size:
                newQBinScale = newQBinScale[:-1]
            newQBinScale[newQBinScale == 0] = 1

            # Compute QBin_vector_expand
            QBin_vector_expand = QBin_vector[newQBinScale - 1]

            # Normalize candidate_distribution_Q
            candidate_distribution_Q = QBin_vector_expand
            sum_Q = np.sum(candidate_distribution_Q)
            if sum_Q > 0.0:
                candidate_distribution_Q_norm = candidate_distribution_Q / sum_Q
            else:
                candidate_distribution_Q_norm = candidate_distribution_Q  # Avoid division by zero

            # Prevent division by zero
            reference_distribution_P_norm[reference_distribution_P_norm == 0] = np.finfo(float).eps
            candidate_distribution_Q_norm[candidate_distribution_Q_norm == 0] = np.finfo(float).eps

            # Compute KL divergence
            Pa = reference_distribution_P_norm
            Pb = np.log(reference_distribution_P_norm / candidate_distribution_Q_norm)
            KLEntropy[iBin-1] = np.sum(Pa * Pb)

            # Store QBinScale
            QScaleVector[iBin-1] = QBinScale

        # Find the minimum KL divergence and its index
        minKLValue = np.nanmin(KLEntropy)
        minKLValueIdx = np.nanargmin(KLEntropy)
        minKL_threshold = curDistributionEdge_P[1 + minKLValueIdx]
        minKL_QScale = QScaleVector[minKLValueIdx]

        # Get reference_distribution_P_norm, candidate_distribution_Q_norm, and reference_distribution_P_edge
        # corresponding to minKLValueIdx
        reference_distribution_P_norm_final = curDistributionN_P[:minKLValueIdx].copy()
        if minKLValueIdx + 1 < len(curDistributionN_P):
            outlier_count = np.sum(curDistributionN_P[minKLValueIdx + 1:])
            reference_distribution_P_norm_final[-1] += outlier_count
        sum_P_final = np.sum(reference_distribution_P_norm_final)
        if sum_P_final > 0.0:
            reference_distribution_P_norm_final = reference_distribution_P_norm_final / sum_P_final
        else:
            reference_distribution_P_norm_final = reference_distribution_P_norm_final  # Avoid division by zero

        reference_distribution_P_edge_final = curDistributionEdge_P[:minKLValueIdx + 1]


        # Recalculate QBinScale and related distributions
        QBinScale_final = QScaleVector[minKLValueIdx]
        RBinScale_final = np.round(reference_distribution_P_edge_final * QBinScale_final).astype(int)[1:]
        RBinScale_final[RBinScale_final == 0] = 1

        # Call the Numba-optimized helper function to compute QBin_vector
        QBin_vector_final = compute_QBin_vector(reference_distribution_P_norm_final, RBinScale_final, maxIntBin)

        # Handle NaN values
        QBin_vector_final = np.nan_to_num(QBin_vector_final, nan=0.0)

        # Compute newQBinScale
        invQBinScale_final = maxIntBin / len(reference_distribution_P_norm_final)
        newQBinScale_final = np.round(np.arange(0, maxIntBin, invQBinScale_final)).astype(int)
        if newQBinScale_final.size != reference_distribution_P_norm_final.size:
            newQBinScale_final = newQBinScale_final[:-1]
        newQBinScale_final[newQBinScale_final == 0] = 1

        # Compute QBin_vector_expand
        QBin_vector_expand_final = QBin_vector_final[newQBinScale_final - 1]

        # Normalize candidate_distribution_Q
        candidate_distribution_Q_final = QBin_vector_expand_final
        sum_Q_final = np.sum(candidate_distribution_Q_final)
        if sum_Q_final > 0.0:
            candidate_distribution_Q_norm_final = candidate_distribution_Q_final / sum_Q_final
        else:
            candidate_distribution_Q_norm_final = candidate_distribution_Q_final  # Avoid division by zero

        # Prevent division by zero
        reference_distribution_P_norm_final[reference_distribution_P_norm_final == 0] = np.finfo(float).eps
        candidate_distribution_Q_norm_final[candidate_distribution_Q_norm_final == 0] = np.finfo(float).eps

        # xlim = max(curDistributionEdge_P)
        # ymax = max(max(reference_distribution_P_norm), max(candidate_distribution_Q_final))
        # dot_figure(reference_distribution_P_norm, curDistributionEdge_P, xlim, ymax, minKL_threshold, ifold, iLayer, "origin")
        # dot_figure(candidate_distribution_Q_final, reference_distribution_P_edge_final, xlim, ymax, minKL_threshold, ifold, iLayer, "final")

        return (
            minKLValue,
            minKLValueIdx,
            minKL_threshold,
            minKL_QScale,
            KLEntropy,
            reference_distribution_P_norm_final,
            candidate_distribution_Q_norm_final,
            reference_distribution_P_edge_final
        )

    @staticmethod
    def create_KL_layer(features_dict, iLayer, Qmax, nRBin, startSearchPoint, ifold):
        # Call findOptimalThreshold to compute necessary data
        (minKLValue, minKLValueIdx, minKL_threshold, minKL_QScale,
         KLEntropy, reference_distribution_P_norm, candidate_distribution_Q_norm, reference_distribution_P_edge) = \
            FindThreshold.findOptimalThreshold(features_dict, iLayer, Qmax, nRBin, startSearchPoint, ifold)

        # Store the computed data into the KL_Layers class dictionary
        layer_key = f"KL_Layer_{iLayer}"
        FindThreshold.KL_Layers[layer_key] = {
            "minKLValue": minKLValue,
            "minKLValueIdx": minKLValueIdx,
            "minKL_threshold": minKL_threshold,
            "minKL_QScale": minKL_QScale,
            "KLEntropy": KLEntropy,
            "reference_distribution_P_norm": reference_distribution_P_norm,
            "candidate_distribution_Q_norm": candidate_distribution_Q_norm,
            "reference_distribution_P_edge": reference_distribution_P_edge
        }

        # Update QAct_params
        FindThreshold.QAct_params[f'conv{iLayer}_activation_Qscale'] = minKL_QScale

        return FindThreshold.KL_Layers[layer_key]

