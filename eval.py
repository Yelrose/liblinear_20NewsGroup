import os
import sys
import numpy as np

def load_file(path):
    vec = []
    fp =  open(path)
    for t in fp.readlines():
        vec.append(int(t.strip().split()[0]))
    return vec


if __name__ == "__main__":
    test_file = sys.argv[1]
    log_file = sys.argv[2]
    test = np.array(load_file(test_file))
    log =  np.array(load_file(log_file))
    accruancy  = 1.0* np.sum(test == log) / len(test)
    macro_precision = 0.
    macro_recall = 0.
    macro_F = 0.
    tp = {}
    fp = {}
    fn = {}
    for i in xrange(len(test)):
        if test[i] not in tp:
            tp[test[i]] = 0.
        if test[i] not in fp:
            fp[test[i]] = 0.
        if test[i] not in fn:
            fn[test[i]] = 0.
    for i in xrange(len(test)):
        if test[i] == log[i]:
            tp[test[i]] += 1.
        if test[i] !=  log[i]:
            fp[test[i]] += 1.
            fn[log[i]] += 1.
    for i in tp:
        precision = 1.0 * tp[i] / (tp[i] + fn[i])
        recall = 1.0 * tp[i] / (tp[i] + fp[i])
        F = 2* precision * recall / (precision + recall)
        macro_precision += precision
        macro_recall += recall
        macro_F += F
    macro_precision /= len(tp)
    macro_recall /= len(tp)
    macro_F /= len(tp)
    print "Accruancy:",accruancy
    print "Precision:",macro_precision
    print "Recall:",macro_recall
    print "F:",macro_F




