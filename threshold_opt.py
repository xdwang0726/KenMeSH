import argparse
import pickle
import numpy as np

def create_score_per_class(P_score, _N, _n):
    scores_per_class = {}  # it will hold predicted score for each class in incresing order
    for i in range(_N):
        scores_per_class[i] = []
        prev = 0
        for j in range(_n):
            prob = (P_score[j][
                        i] + prev) / 2.0  # if we want to use posible value for probability (section 2 in paper): (t k = s k (x (i) ) + s k (x (i+1) ) /2)
            scores_per_class[i].append(prob)
            prev = P_score[j][i]
            # scores_per_class[i].append(P_score[j][i])
        scores_per_class[i].sort()
    return scores_per_class


def calculateF(P_score, T_score, T, _N, _n, beta=1):  # return F1-score for any given ThreshHold tensor
    '''
        input threshhold tensor and beta value
        Micro-averaging F-score in the paper eq 5,6
        O(n * N)
    '''
    true_pos = []  # true positive count per class
    false_pos = []  # false positive count per class
    false_neg = []  # false negative count per class
    A = 0
    B = 0
    C = 0
    D = 0

    for x in range(_N):
        tp = 0
        fp = 0
        fn = 0
        for i in range(_n):
            pred = 0
            if T[x] <= P_score[i][x]:
                pred = 1
            if pred == 1 and T_score[i][x] == 1:
                tp += 1
            if pred == 1 and T_score[i][x] == 0:
                fp += 1
            if pred == 0 and T_score[i][x] == 1:
                fn += 1
        # print(tp," ", fp , " " , fn)
        A += tp
        C += tp
        B += (tp + fp)
        D += (tp + fn)
    precision = A / B
    recall = C / D

    f_score = (1 + beta * beta) / ((1.0 / precision) + (beta * beta) / recall)
    return f_score, A, B, C, D


def updated_score_T(P_score, T_score, _n, k, curT, prevT, beta, pd, pn, rd, rn):
    '''
    updates F-score in O(n)
    '''
    TP = 0
    FP = 0
    FN = 0
    PTP = 0
    PFP = 0
    PFN = 0
    for i in range(_n):
        pred = 0
        if curT <= P_score[i][k]:
            # print("?? ",curT , P_score[i][k],i)
            pred = 1
        if pred == 1 and T_score[i][k] == 1:
            TP += 1
        if pred == 1 and T_score[i][k] == 0:
            FP += 1
        if pred == 0 and T_score[i][k] == 1:
            FN += 1
        pred = 0
        if prevT <= P_score[i][k]:
            pred = 1
        if pred == 1 and T_score[i][k] == 1:
            PTP += 1
        if pred == 1 and T_score[i][k] == 0:
            PFP += 1
        if pred == 0 and T_score[i][k] == 1:
            PFN += 1
    # print("stat for class",k)
    # print("prev : ",prevT, PTP, PFP, PFN)
    # print("cur : ", curT, TP, FP, FN)
    # print("pd : ",pd,pn,rd,rn)
    A = pd - PTP + TP
    B = (pn - (PTP + PFP) + (TP + FP))
    precision = A / B
    C = rd - PTP + TP
    D = rn - (PTP + PFN) + (TP + FN)
    recall = C / D
    f_score = (1 + beta * beta) / ((1.0 / precision) + (beta * beta) / recall)

    return f_score, A, B, C, D


def find_arg_max(P_score, T_score, _n, scores_per_class, k, t, beta, curF, pd, pn, rd, rn):
    '''
    Finds armax t for any given class k
    in O(N * n) // as there are O(N) different threshold value and for every possible threshold value we need to see
                whether we can update the f-score using updated_score_T()
    '''
    poss_value_for_t = scores_per_class[k]
    # curF,_,_,_,_ = calculateF(t,beta)
    res = t[k]
    for x in poss_value_for_t:  # probably can be improve using binary search
        if x >= t[k]:
            tmp_fscore, a, b, c, d = updated_score_T(P_score, T_score, _n, k, x, t[k], beta, pd, pn, rd, rn)
            if curF < tmp_fscore:
                curF = tmp_fscore
                pd = a
                pn = b
                rd = c
                rn = d
                res = x
                t[k] = x
            # t[k] = tmp_t
    # print("class",k,res,curF,pd,pn,rd,rn)
    return [res, curF, pd, pn, rd, rn]


def maximization_Algo1(P_score, T_score, scores_per_class, _N, _n, maximum_iteration):  # will return the threshhold
    '''
        According to the paper : the iterative improvment process will coverge
        However, Total run-time for Iteration : O(N * N * n)
        The paper proves the overall run-time to be O(n^2 * N^2) [Appendix D]
        This run-time will not be feasible for larger data-sets (number of class and number of data point)
        However, as the improvment is increamental may be we can treat the iteration as a hyper-parameter.
    '''
    # initialize t
    t = [0.35] * _N
    beta = 1
    # for i in range(_N):
    #     t.append(scores_per_class[i][0])  # assuming we have at least one example with class i
    iter = 0
    curF, precd, precsum, recalld, recallsum = calculateF(P_score, T_score, t, _N, _n, beta)
    print("Init F: ", curF)
    while (1):
        if iter % 100 == 0:
            print("Iteration ", iter)
        is_updated = False
        for k in range(_N):
            tmp_t, tmp_fscore, a, b, c, d = find_arg_max(P_score, T_score, _n, scores_per_class, k, t, beta, curF,
                                                         precd, precsum, recalld, recallsum)
            # print(tmp_t," C, ",a," ",b," ",c," ",d)
            if tmp_fscore > curF:
                curF = tmp_fscore
                precd = a
                precsum = b
                recalld = c
                recallsum = d
                t[k] = tmp_t
                is_updated = True
                break

        if is_updated == False:
            break
        iter += 1
        if iter == maximum_iteration:
           break
    return t, curF


def eval(y_true, y_pred, num_labels, num_test):
    scores_per_class = create_score_per_class(y_pred, num_labels, num_test)
    t, imp_F = maximization_Algo1(y_pred, y_true, scores_per_class, num_labels, num_test)
    micro_f_score, A, B, C, D = calculateF(y_pred, y_true, t, num_labels, num_test, beta=1)
    micro_precision = A / B
    micro_recall = C / D
    return micro_precision, micro_recall, micro_f_score


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted')
    parser.add_argument('--true')
    parser.add_argument('--save')
    args = parser.parse_args()

    P_score = pickle.load(open(args.predicted, 'rb'))
    P_score = np.concatenate(P_score, axis=0)
    T_score = pickle.load(open(args.true, 'rb'))
    T_score = np.concatenate(T_score, axis=0)
    _N = len(P_score[0])
    _n = len(P_score)
    maximum_iteration = 3
    scores_per_class = create_score_per_class(P_score, _N, _n)
    t, imp_F = maximization_Algo1(P_score, T_score, scores_per_class, _N, _n, maximum_iteration)
    print(imp_F)
    pickle.dump(t, open(args.save, 'wb'))


if __name__ == "__main__":
    main()