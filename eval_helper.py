import numpy as np
from scipy import stats
from scipy.sparse import issparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


def zero_division(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def precision(p, t):
    """
    p, t: two sets of labels/integers
    >>> precision({1, 2, 3, 4}, {1})
    0.25
    """
    return len(t.intersection(p)) / len(p)


def precision_at_ks(Y_pred_scores, Y_test, ks):
    """
    Y_pred_scores: nd.array of dtype float, entry ij is the score of label j for instance i
    Y_test: list of label ids
    """
    result = []
    for k in ks:
        Y_pred = []
        for i in np.arange(Y_pred_scores.shape[0]):
            if issparse(Y_pred_scores):
                idx = np.argsort(Y_pred_scores[i].data)[::-1]
                Y_pred.append(set(Y_pred_scores[i].indices[idx[:k]]))
            else:  # is ndarray
                idx = np.argsort(Y_pred_scores[i, :])[::-1]
                Y_pred.append(set(idx[:k]))

        result.append([precision(yp, set(yt)) for yt, yp in zip(Y_test, Y_pred)])
    return result


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


# def perf_measure(y_actual, y_hat):
#     TP_total = []
#     FP_total = []
#     TN_total = []
#     FN_total = []
#
#     for i in range(y_actual.shape[1]):
#         TP = 1
#         FP = 1
#         TN = 1
#         FN = 1
#
#         for j in range(y_actual.shape[0]):
#             if y_actual[j, i] == y_hat[j, i] == 1:
#                 TP += 1
#             if y_hat[j, i] == 1 and y_actual[j, i] != y_hat[j, i]:
#                 FP += 1
#             if y_actual[j, i] == y_hat[j, i] == 0:
#                 TN += 1
#             if y_hat[j, i] == 0 and y_actual[j, i] != y_hat[j, i]:
#                 FN += 1
#         TP_total.append(TP)
#         FP_total.append(FP)
#         TN_total.append(TN)
#         FN_total.append(FN)
#
#     MaP = macro_precision(TP_total, FP_total)
#     MiP = micro_precision(TP_total, FP_total)
#     MaR = macro_recall(TP_total, FN_total)
#     MiR = micro_recall(TP_total, FN_total)
#     MaF = macro_f1(MaP, MaR)
#     MiF = micro_f1(MiP, MiR)
#
#     result = [round(MaP, 5), round(MiP, 5), round(MaF, 5), round(MiF, 5)]
#     return result


# def micro_macro_eval(y_actual, y_hat):
#     MaP = precision_score(y_actual, y_hat, average='macro')
#     MiP = precision_score(y_actual, y_hat, average='micro')
#     MaR = recall_score(y_actual, y_hat, average='macro')
#     MiR = recall_score(y_actual, y_hat, average='micro')
#     MaF = f1_score(y_actual, y_hat, average='macro')
#     MiF = f1_score(y_actual, y_hat, average='micro')
#
#     result = [round(MaP, 5), round(MiP, 5), round(MaR, 5), round(MiR, 5), round(MaF, 5), round(MiF, 5)]
#    return result


def example_based_precision(CL, y_hat):
    EBP = []

    for i in range(len(CL)):
        ebp = zero_division(CL[i], len(y_hat[i]))
        EBP.append(ebp)

    EBP = np.mean(EBP)
    return EBP


def example_based_recall(CL, y_actural):
    EBR = []

    for i in range(len(CL)):
        ebr = zero_division(CL[i], len(y_actural[i]))
        EBR.append(ebr)
    EBR = np.mean(EBR)
    return EBR


def example_based_fscore(CL, y_actual, y_hat):
    EBF = []

    for i in range(len(CL)):
        ebf = zero_division((2 * CL[i]), (len(y_hat[i]) + len(y_actual[i])))
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


# def example_based_evaluation(y_actual, y_hat):
#     num_common_label = find_common_label(y_actual, y_hat)
#
#     EBP = example_based_precision(num_common_label, y_hat)
#     EBR = example_based_recall(num_common_label, y_actual)
#     EBF = example_based_fscore(num_common_label, y_actual, y_hat)
#     result = [round(EBP, 5), round(EBR, 5), round(EBF, 5)]
#     return result

# def evalution(pred, true_label):
#     return
#
#
# """
# def ranking_precision_score(y_true, y_score, k=10):
# """Precision at rank k
#     Parameters
#     ----------
#     y_true : array-like, shape = [n_samples]
#         Ground truth (true relevance labels).
#     y_score : array-like, shape = [n_samples]
#         Predicted scores.
#     k : int
#         Rank.
#     Returns
#     -------
#     precision @k : float
#     """
# unique_y = np.unique(y_true)
# if len(unique_y) > 2:
# raise ValueError("Only supported for two relevance levels.")
# pos_label = unique_y[1]
# n_pos = np.sum(y_true == pos_label)
# order = np.argsort(y_score)[::-1]
# y_true = np.take(y_true, order[:k])
# n_relevant = np.sum(y_true == pos_label)
# # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
# return float(n_relevant) / min(n_pos, k)
# """


def example_based_evaluation(pred, target, threshold):
    pred = np.greater_equal(pred, threshold).astype(np.int)

    common_label = np.sum(np.multiply(pred, target), axis=1)
    sum_pred = np.sum(pred, axis=1)
    sum_true = np.sum(target, axis=1)


    ebp = np.average(np.sum(common_label / sum_pred))
    ebr = np.average(np.sum(common_label / sum_true))
    ebf = 2 * ebp * ebr / (ebp + ebr)

    return (ebp, ebr, ebf)


def micro_macro_eval(pred, target, threshold):
    positive = 1
    negative = 0

    pred = np.greater_equal(pred, threshold).astype(np.int)

    tp = np.logical_and(pred == positive, target == positive).astype(np.int)
    tn = np.logical_and(pred == negative, target == negative).astype(np.int)
    fp = np.logical_and(pred == positive, target == negative).astype(np.int)
    fn = np.logical_and(pred == negative, target == positive).astype(np.int)

    sum_tp = np.sum(tp)
    sum_tn = np.sum(tn)
    sum_fp = np.sum(fp)
    sum_fn = np.sum(fn)

    MiP = sum_tp / (sum_tp + sum_fp)
    MiR = sum_tp / (sum_tp + sum_fn)
    MiF = 2 * MiP * MiR / (MiP + MiR)

    MaP = np.average(np.nan_to_num(np.divide(np.sum(tp, axis=0), (np.sum(tp, axis=0) + np.sum(fp, axis=0)))))
    MaR = np.average(np.nan_to_num(np.divide(np.sum(tp, axis=0), (np.sum(tp, axis=0) + np.sum(fn, axis=0)))))
    MaF  = 2 * MaP * MaR / (MaP + MaR)

    return (MiF, MiP, MiR, MaF, MaP, MaR)


