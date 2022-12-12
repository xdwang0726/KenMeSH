import numpy as np
from scipy.sparse import issparse


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


def recall(p, t):
    return len(t.intersection(p)) / len(t)


def precision_at_ks(Y_pred_scores, Y_test, ks):
    """
    Y_pred_scores: nd.array of dtype float, entry ij is the score of label j for instance i
    Y_test: list of label ids
    """
    p = []
    r = []
    for k in ks:
        Y_pred = []
        arr = np.array(Y_pred_scores)
        for i in np.arange(arr.shape[0]):
            if issparse(arr):
                idx = np.argsort(arr[i].data)[::-1]
                Y_pred.append(set(arr[i].indices[idx[:k]]))
            else:  # is ndarray
                idx = np.argsort(arr[i, :])[::-1]
                Y_pred.append(set(idx[:k]))

        p.append([precision(yp, set(yt)) for yt, yp in zip(Y_test, Y_pred)])
        r.append([recall(yp, set(yt)) for yt, yp in zip(Y_test, Y_pred)])
    return p, r


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


def example_based_evaluation(pred, target, threshold, num_example):
    pred = np.greater_equal(pred, threshold).astype(np.int)

    common_label = np.sum(np.multiply(pred, target), axis=1)
    sum_pred = np.sum(pred, axis=1)
    sum_true = np.sum(target, axis=1)


    ebp = np.sum(np.nan_to_num(common_label / sum_pred)) / num_example
    ebr = np.sum(np.nan_to_num(common_label / sum_true)) / num_example
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
