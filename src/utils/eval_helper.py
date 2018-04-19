import numpy as np

def knn_accuracy_from_matrix(F_pred, F_gold, k=1, src_words=None, tgt_words=None, verbose=False, train_set=set()):
    total = 0.0
    tp = 0.0
    gold_sum = np.sum(F_gold, axis=1)
    # print(np.sum(F_pred, axis=1))
    # print(gold_sum)
    for i in range(F_gold.shape[0]):
        if i in train_set:
            continue
        if gold_sum[i] > 0:
            total += 1.0
            if verbose:
                print("Query: ", src_words[i])
                print("Gold:")
                for gold_j in np.nonzero(F_gold[i,:])[0].tolist():
                    print("\t" + tgt_words[gold_j])

                print("System:")
            for j in np.argsort(-F_pred[i,:]).tolist()[:k]:
                if verbose:
                    print 'current j', j, F_pred.shape
                    print("\t" + tgt_words[j])
                if F_gold[i,j] == 1:
                    tp += 1.0
    print("evaluated on %d pairs" % (total))
    return tp/total