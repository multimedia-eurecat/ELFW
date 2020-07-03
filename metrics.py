# This file computes common semantic segmentation metrics
# Rafael Redondo and Jaume Gibert (c) Eurecat 2019

import numpy as np
e = 1E-10   # epsilon

def ZerosTFPN(num_classes):

    return np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)


def TrueFalsePositiveNegatives(labels, predictions, num_classes):

    TP, TN, FP, FN = ZerosTFPN(num_classes)

    for c in range(0, num_classes):

        A = (predictions == c).cpu().detach().numpy()
        B = (labels == c).cpu().detach().numpy()
        C = np.logical_not(A)
        D = np.logical_not(B)

        TP[c] = np.sum(np.logical_and(A, B))  # True Positives
        TN[c] = np.sum(np.logical_and(C, D))  # True Negatives
        FP[c] = np.sum(np.logical_and(A, D))  # False Positives
        FN[c] = np.sum(np.logical_and(C, B))  # False Negatives

    return TP, TN, FP, FN


def PixelAccuracy(TP, FN):

    return np.sum(TP) / (np.sum(TP + FN) + e)


def MeanAccuracy(TP, FN):   # Also True Positive Rate (TPR) on average for all classes

    accuracy = TP / (TP + FN + e)
    return np.mean(accuracy), accuracy


def MeanIU(TP, FN, FP):     # Also Threat Score(TS) or Critical Success Index (CSI)

    iu = TP / (TP + FN + FP + e)
    return np.mean(iu), iu


def FrequencyWeightedIU(TP, FN, FP):

    total_i = TP + FN
    return np.sum(total_i * TP / (total_i + FP + e)) / (np.sum(total_i) + e)


def MeanF1Score(TP, FN, FP):

    f1_score = 2*TP / (2*TP + FP + FN + e)
    return np.mean(f1_score), f1_score
