import numpy as np
from scipy import stats


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def macro_precision(TP, FP):
    MaP = []
    for i in range(len(TP)):
        macro_p = TP[i] / (TP[i] + FP[i])
        MaP.append(macro_p)
    MaP = np.mean(MaP)
    return MaP


def macro_recall(TP, FN):
    MaR = []
    for i in range(len(TP)):
        macro_r = TP[i] / (TP[i] + FN[i])
        MaR.append(macro_r)
    MaR = np.mean(MaR)
    return MaR


def micro_precision(TP, FP):
    MiP = sum(TP) / (sum(TP) + sum(FP))
    return MiP


def micro_recall(TP, FN):
    MiR = sum(TP) / (sum(TP) + sum(FN))
    return MiR


def macro_f1(MaP, MaR):
    MaF = stats.hmean([MaP, MaR])
    return MaF


def micro_f1(MiP, MiR):
    MiF = stats.hmean([MiP, MiR])
    return MiF


def perf_measure(y_actual, y_hat):
    TP_total = []
    FP_total = []
    TN_total = []
    FN_total = []

    for i in range(y_actual.shape[1]):
        TP = 1
        FP = 1
        TN = 1
        FN = 1

        for j in range(y_actual.shape[0]):
            if y_actual[j, i] == y_hat[j, i] == 1:
                TP += 1
            if y_hat[j, i] == 1 and y_actual[j, i] != y_hat[j, i]:
                FP += 1
            if y_actual[j, i] == y_hat[j, i] == 0:
                TN += 1
            if y_hat[j, i] == 0 and y_actual[j, i] != y_hat[j, i]:
                FN += 1
        TP_total.append(TP)
        FP_total.append(FP)
        TN_total.append(TN)
        FN_total.append(FN)

    MaP = macro_precision(TP_total, FP_total)
    MiP = micro_precision(TP_total, FP_total)
    MaR = macro_recall(TP_total, FN_total)
    MiR = micro_recall(TP_total, FN_total)
    MaF = macro_f1(MaP, MaR)
    MiF = micro_f1(MiP, MiR)

    result = [round(MaP, 5), round(MiP, 5), round(MaF, 5), round(MiF, 5)]
    return result


def example_based_precision(CL, y_hat):
    EBP = []

    for i in range(len(CL)):
        ebp = CL[i] / len(y_hat[i])
        EBP.append(ebp)

    EBP = np.mean(EBP)
    return EBP


def example_based_recall(CL, y_actural):
    EBR = []

    for i in range(len(CL)):
        ebr = CL[i] / len(y_actural[i])
        EBR.append(ebr)
    EBR = np.mean(EBR)
    return EBR


def example_based_fscore(CL, y_actual, y_hat):
    EBF = []

    for i in range(len(CL)):
        ebf = (2 * CL[i]) / (len(y_hat[i]) + len(y_actual[i]))
        EBF.append(ebf)

    EBF = np.mean(EBF)
    return EBF


def find_common_label(y_actual, y_hat):
    num_common_label = []

    for i in range(len(y_actual)):
        labels = intersection(y_actual[i], y_hat[i])
        num_label = len(labels)
        num_common_label.append(num_label)
    return num_common_label


def example_based_evaluation(y_actual, y_hat):
    num_common_label = find_common_label(y_actual, y_hat)

    EBP = example_based_precision(num_common_label, y_hat)
    EBR = example_based_recall(num_common_label, y_actual)
    EBF = example_based_fscore(num_common_label, y_actual, y_hat)
    result = [round(EBP, 5), round(EBR, 5), round(EBF, 5)]
    return result
