import numpy as np
import pandas as pd
import sys
from keras import backend as k
import tensorflow as tf
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, roc_auc_score, average_precision_score
sys.path.append('src')
from generate_structure import MODEL_11_OUT_FIRST_STAGE, MODEL_11_OUT_SECOND_STAGE, TEST_ANNOTATIONS


def accuracy_ml(y_pred,y_true):
    y = y_true == 1

    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]

    acc = 0
    for i in range(len(y_t)):
        s,t = set(y_t[i]),set(y_p[i])
        union = s.union(t)
        intersection = s.intersection(t)
        acc += (len(intersection)/len(union))
    return acc/len(y_t)


def precision_ml(y_pred,y_true):
    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]
    
    prec = 0
    for i in range(len(y_t)):
        s,t = set(y_t[i]),set(y_p[i])
        intersection = s.intersection(t)
        if(len(t) != 0):
            prec += (len(intersection)/len(t))
    return prec/len(y_t)    


def recall_ml(y_pred,y_true):
    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]
 
    recall = 0

    for i in range(len(y_t)):
        s,t = set(y_t[i]),set(y_p[i])
        intersection = s.intersection(t)
        if(len(s) != 0):
            recall += (len(intersection)/len(s))
    return recall/len(y_t)


def metrics_cnn():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "y_proba_role_sum_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "y_pred_role_sum_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred,y_test)
    acc_score = accuracy_score(y_pred,y_test)
    acc_ml = accuracy_ml(y_pred,y_test)
    prec_ml = precision_ml(y_pred,y_test)
    rec_ml = recall_ml(y_pred,y_test)
    fmeasure_score = f1_score(y_pred,y_test, average='micro')
    roc_score = roc_auc_score(y_test,y_proba, average='micro')
    pr_score = average_precision_score(y_test,y_proba,average='micro')

    print('{0:.4f};{1:.4f};{2:.4f};{3:.4f};{4:.4f};{5:.4f};{6:.4f};{7:.4f};'.format(
        hl_score, acc_score, acc_ml, fmeasure_score, roc_score, pr_score, prec_ml, rec_ml))


if __name__ == '__main__':
    metrics_cnn()
    
