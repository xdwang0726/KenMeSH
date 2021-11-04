from tqdm import tqdm

# Algorithm from the paper : Threshold optimization for multi-label classifieres

maximum_iteration = 10


def create_score_per_class(_N, _n, P_score):
    """it will hold predicted score for each class in incresing order"""
    scores_per_class = {}
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


def calculateF(_N, _n, P_score, T_score, T, beta=1):  # return F1-score for any given ThreshHold tensor
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


def updated_score_T(_n, P_score, T_score, k, curT, prevT, beta, pd, pn, rd, rn):
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

    A = pd - PTP + TP
    B = (pn - (PTP + PFP) + (TP + FP))
    precision = A / B
    C = rd - PTP + TP
    D = rn - (PTP + PFN) + (TP + FN)
    recall = C / D
    f_score = (1 + beta * beta) / ((1.0 / precision) + (beta * beta) / recall)

    return f_score, A, B, C, D


def find_arg_max(_N, _n, P_score, T_score, k, t, beta, curF, pd, pn, rd, rn):
    '''
    Finds armax t for any given class k
    in O(N * n) // as there are O(N) different threshold value and for every possible threshold value we need to see
                whether we can update the f-score using updated_score_T()
    '''
    scores_per_class = create_score_per_class(_N, _n, P_score)
    poss_value_for_t = scores_per_class[k]
    res = t[k]
    for x in poss_value_for_t:  # probably can be improve using binary search
        if x >= t[k]:
            tmp_fscore, a, b, c, d = updated_score_T(_n, P_score, T_score, k, x, t[k], beta, pd, pn, rd, rn)
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


def maximization_Algo1(_N, _n, P_score, T_score):  # will return the threshhold
    '''
        According to the paper : the iterative improvment process will coverge
        However, Total run-time for Iteration : O(N * N * n)
        The paper proves the overall run-time to be O(n^2 * N^2) [Appendix D]
        This run-time will not be feasible for larger data-sets (number of class and number of data point)
        However, as the improvment is increamental may be we can treat the iteration as a hyper-parameter.
    '''
    t = [5e-4] * _N
    beta = 1
    iter = 0
    curF, precd, precsum, recalld, recallsum = calculateF(_N, _n, P_score, T_score, t, beta)
    print("Init F: ", curF)
    while (1):
        if iter % 100 == 0:
            print("Iteration ", iter)
        is_updated = False
        for k in tqdm(range(_N)):
            # print(curF,precd,precsum,recalld,recallsum)
            # print(t)
            # print(k)
            tmp_t, tmp_fscore, a, b, c, d = find_arg_max(_N, _n, P_score, T_score, k, t, beta, curF, precd, precsum, recalld, recallsum)
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


def get_threshold(_N, _n, P_score, T_score):
    t, imp_F = maximization_Algo1(_N, _n, P_score, T_score)
    print("F: ", calculateF(_N, _n, P_score, T_score, t))
    return t

