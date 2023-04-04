import numpy as np
from scipy import stats
from scipy.sparse import issparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import torch
from torch.utils.data import DataLoader, random_split
import logging
from check_label import get_index_to_meshid

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
    true_meshes = []
    probs = []

    for i in t: 
        true_meshes.append(get_index_to_meshid(indx=i))
    for j in p:
        probs.append(get_index_to_meshid(indx=j))

    print("p,t: ", p,t)
    print("probsss,truesss: ", probs,true_meshes)
    
    return len(t.intersection(p)) / len(p)

def precision_at_ks(Y_pred_scores, Y_test, ks):
    """
    Y_pred_scores: nd.array of dtype float, entry ij is the score of label j for instance i
    Y_test: list of label ids
    """
    # print("Test 2: ", Y_test[0])
    result = []
    for k in ks:
        Y_pred = []
        for i in np.arange(Y_pred_scores.shape[0]):
            if issparse(Y_pred_scores):
                idx = np.argsort(Y_pred_scores[i].data)[::-1]
                Y_pred.append(set(Y_pred_scores[i].indices[idx[:k]]))
            else:  # is ndarray
                idx = np.argsort(Y_pred_scores[i, :])[::-1]
                # [i, :] : picks all columns of ith row, specific to numpy
                # np.argsort returns the indices of the sorted array
                # [::-1] arranges the array in a descending order
                # print("1: ",  Y_pred_scores)
                # print("2: ",  Y_pred_scores[i, :])
                # print("3: ",  np.argsort(Y_pred_scores[i, :]))
                # print("4: ",  np.argsort(Y_pred_scores[i, :])[::-1])
                # print("5: ",  np.argsort(Y_pred_scores[i, :][::-1]))
                Y_pred.append(set(idx[:k]))

        # print("precision_at_ks Y_pred: ", Y_pred)
        # print("precision_at_ks Y_test: ", Y_test[0])
        result.append(np.mean([precision(yp, set(yt)) for yt, yp in zip(Y_test, Y_pred)]))
    return result

def precision_at_k(y_true, y_pred, k):
    """
    Calculate precision at k for a list of true labels and predicted labels.

    Args:
        y_true: numpy array of true labels (shape: [n_samples, n_labels])
        y_pred: numpy array of predicted labels (shape: [n_samples, n_labels])
        k: integer specifying the position to calculate precision at (1, 3, or 5)

    Returns:
        precision: float precision at k
    """
    n_samples, n_labels = y_true.shape
    precision = 0.0
    for i in range(n_samples):
        true_labels = y_true[i]
        pred_labels = y_pred[i]
        sorted_indices = np.argsort(pred_labels)[::-1] # sort in descending order
        top_k = sorted_indices[:k]
        correct_labels = np.sum(true_labels[top_k])
        precision += correct_labels / k
    precision /= n_samples
    return precision

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


def example_based_evaluation(pred, target, num_example):
    # pt = [(i,m) for i, m in enumerate(pred[0]) if m > 4.1633608e-05]
    # pred = np.greater_equal(pred, threshold).astype(int)
    # p = [(i,m) for i, m in enumerate(pred[0]) if m != 0]
    print("Example evaluation pred: ", pred, pred.shape)
    print("Example evaluation target: ", target, target.shape)
    # print(" target count: ", [m for m in target[0] if m != 0])
    # print(" pred count: ", [m for m in pred[0] if m != 0])

    common_label = np.sum(np.multiply(pred, target), axis=1)
    sum_pred = np.sum(pred, axis=1)
    print("common_label: ", common_label)
    print("sum_pred: ", sum_pred)
    sum_true = np.sum(target, axis=1)
    print("example_based_evaluation: ")
    print(common_label, sum_pred, num_example, sum_true)
    try:    
        ebp = np.sum(np.nan_to_num(common_label / sum_pred)) / num_example
        ebr = np.sum(np.nan_to_num(common_label / sum_true)) / num_example
        ebf = 2 * ebp * ebr / (ebp + ebr)
    except BaseException as exception:
            logging.warning(f"Exception Name: {type(exception).__name__}")
            logging.warning(f"Exception Desc: {exception}")
    return (ebp, ebr, ebf)


def micro_macro_eval(pred, target, threshold):
    positive = 1
    negative = 0

    pred = np.greater_equal(pred, threshold).astype(int)

    tp = np.logical_and(pred == positive, target == positive).astype(int)
    tn = np.logical_and(pred == negative, target == negative).astype(int)
    fp = np.logical_and(pred == positive, target == negative).astype(int)
    fn = np.logical_and(pred == negative, target == positive).astype(int)

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


def getLabelIndex(labels):
    label_index = np.zeros((len(labels), len(labels[1])))
    for i in range(0, len(labels)):
        index = np.where(labels[i] == 1)
        index = np.asarray(index)
        N = len(labels[1]) - index.size
        index = np.pad(index, [(0, 0), (0, N)], 'constant')
        label_index[i] = index

    label_index = np.array(label_index, dtype=int)
    label_index = label_index.astype(int)
    return label_index

def flatten(l):
    flat = [i for item in l for i in item]
    return flat

def main():
    label_test = pickle.load(open("label_test.pkl", 'rb'))
    label_test = np.array(label_test)
    P_score = torch.load("pred2")
    P_score = np.concatenate(P_score, axis=0) # 3d -> 2d array

    probsss = 1 / (1 + np.exp(P_score))
    preds_tensor = torch.tensor(P_score)
    preds_probs_t = torch.sigmoid(preds_tensor)
    preds_probs = preds_probs_t.numpy()

    # Convert the negative log probabilities to probabilities using softmax
    probs = torch.softmax(-preds_tensor, dim=1)

    # Replace NaNs with zeros (this can happen if there are any -inf values in pred_labels)
    probs[torch.isnan(probs)] = 0

    # Convert the probabilities to binary predictions using a threshold (e.g. 0.5)
    bin_pred_labels = (probs > 0.0005).int()
    bin_pred_labels_np = bin_pred_labels.numpy()

    # print("Pred load done", type(P_score), P_score)
    T_score = torch.load('true_label2')
    T_score = np.concatenate(T_score, axis=0)
    # T_score = T.numpy()
    # print("T_score", type(T_score), len(T_score), T_score, T_score.shape)
    # print("P_score", type(P_score), len(P_score), P_score, P_score.shape)
    # print("Label test load done", type(label_test), len(label_test), label_test, label_test.shape)
    threshold = np.array([1.6170531e-04] * 28415)

    # c = [i for i, m in enumerate(T_score[0]) if m != 0]
    # print("C: ", c)
    test_labelsIndex = getLabelIndex(T_score)
    d = [m for m in test_labelsIndex[0] if m != 0]
    # print("D: ", d, len(d))

    # for i,m in enumerate(T_score[0]):
    #     print(T_score[0][i], test_labelsIndex[0][i])
    
    # print("Eval Helper: test_labelsIndex: ", type(test_labelsIndex), test_labelsIndex.size, test_labelsIndex.shape, test_labelsIndex[0])
    # print("Eval Helper: P_score: ", type(P_score), P_score.size, P_score.shape, P_score[0].shape, np.min(P_score[0]), np.mean(P_score[0]), np.max(P_score[0]))
    # print("Eval Helper: preds_probs: ", type(preds_probs), preds_probs.size, preds_probs.shape, preds_probs[0].shape)
    # print(np.min(preds_probs[0]), np.mean(preds_probs[0]), np.max(preds_probs[0]))
    # print("Eval Helper: bin_pred_labels_np: ", type(bin_pred_labels_np), bin_pred_labels_np.size, bin_pred_labels_np.shape, bin_pred_labels_np[0].shape)
    # print(np.min(bin_pred_labels_np[0]), np.mean(bin_pred_labels_np[0]), np.max(bin_pred_labels_np[0]))

    # flatten the matrix to a 1D array
    # flat_matrix = preds_probs.flatten()

    # sort the array in descending order
    # sorted_matrix = np.sort(flat_matrix)[::-1]

    # get the top 10th max values
    # top_10th_max = sorted_matrix[:int(len(sorted_matrix)*0.05)]

    # print("top_10th_max: ", top_10th_max)
    # print("Test 1: ", test_labelsIndex[0])
    # precisions = precision_at_ks(P_score, test_labelsIndex, ks=[1, 3, 5])
    # print('p@k', precisions)

    precision_1 = precision_at_k(T_score, P_score, 1)
    precision_3 = precision_at_k(T_score, P_score, 3)
    precision_5 = precision_at_k(T_score, P_score, 5)
    print("Precision@1:", precision_1)
    print("Precision@3:", precision_3)
    print("Precision@5:", precision_5)

    y_pred_binary = (preds_probs >= 0.05).astype(int)
    # print(y_pred_binary.shape, y_pred_binary[0])
    emb = example_based_evaluation(y_pred_binary, T_score, len(P_score))
    print('(ebp, ebr, ebf): ', emb)

    c = [i for i, m in enumerate(y_pred_binary[0]) if m != 0]
    # print("C: ", c, len(c))
    print("Intersection: ", intersection(c,d))


    precision = precision_score(T_score, y_pred_binary, average='samples')
    recall = recall_score(T_score, y_pred_binary, average='samples')

    print("precision, recall: ", precision, recall)

    micro = micro_macro_eval(y_pred_binary, T_score, threshold)
    print('mi/ma(MiF, MiP, MiR, MaF, MaP, MaR): ', micro)


if __name__ == "__main__":
    main()