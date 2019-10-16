import pandas as pd
import sys
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, roc_auc_score, average_precision_score
sys.path.append('src')
from generate_structure import MODEL_14_OUT_FIRST_STAGE, MODEL_14_OUT_SECOND_STAGE, TEST_ANNOTATIONS


def accuracy_ml(y_pred,y_true):
    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]

    acc = 0
    for i in range(len(y_t)):
        s,t = set(y_t[i]), set(y_p[i])
        union = s.union(t)
        intersection = s.intersection(t)
        acc += (len(intersection)/len(union))
    return acc/len(y_t)


def precision_ml(y_pred,y_true):
    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]
    
    prec = 0
    for i in range(len(y_t)):
        s,t = set(y_t[i]), set(y_p[i])
        intersection = s.intersection(t)
        if len(t) != 0:
            prec += (len(intersection)/len(t))
    return prec/len(y_t)    


def recall_ml(y_pred, y_true):
    y_t = [[index for index, value in enumerate(data) if value == 1] for data in y_true]
    y_p = [[index for index, value in enumerate(data) if value == 1] for data in y_pred]
 
    recall = 0

    for i in range(len(y_t)):
        s,t = set(y_t[i]), set(y_p[i])
        intersection = s.intersection(t)
        if len(s) != 0:
            recall += (len(intersection)/len(s))
    return recall/len(y_t)


def metrics_cnn_sum_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_pred_role_sum_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_proba_role_sum_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    acc_ml = accuracy_ml(y_pred, y_test)
    prec_ml = precision_ml(y_pred, y_test)
    rec_ml = recall_ml(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_14_OUT_FIRST_STAGE + 'result_sum_late_fusion_stage_1.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


def metrics_cnn_prob_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_pred_role_prob_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_proba_role_prob_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    acc_ml = accuracy_ml(y_pred, y_test)
    prec_ml = precision_ml(y_pred, y_test)
    rec_ml = recall_ml(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_14_OUT_FIRST_STAGE + 'result_prob_late_fusion_stage_1.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


def metrics_cnn_median_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_pred_role_median_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_proba_role_median_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    acc_ml = accuracy_ml(y_pred, y_test)
    prec_ml = precision_ml(y_pred, y_test)
    rec_ml = recall_ml(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_14_OUT_FIRST_STAGE + 'result_median_late_fusion_stage_1.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


def metrics_cnn_max_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_pred_role_max_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "y_proba_role_max_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    acc_ml = accuracy_ml(y_pred, y_test)
    prec_ml = precision_ml(y_pred, y_test)
    rec_ml = recall_ml(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_14_OUT_FIRST_STAGE + 'result_max_late_fusion_stage_1.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


def metrics_cnn_sum_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_pred_role_sum_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_proba_role_sum_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred,y_test)
    acc_score = accuracy_score(y_pred,y_test)
    acc_ml = accuracy_ml(y_pred,y_test)
    prec_ml = precision_ml(y_pred,y_test)
    rec_ml = recall_ml(y_pred,y_test)
    fmeasure_score = f1_score(y_pred,y_test, average='micro')
    roc_score = roc_auc_score(y_test,y_proba, average='micro')
    pr_score = average_precision_score(y_test,y_proba,average='micro')

    with open(MODEL_14_OUT_SECOND_STAGE + 'result_sum_late_fusion_stage_2.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


def metrics_cnn_prob_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_pred_role_prob_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_proba_role_prob_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    acc_ml = accuracy_ml(y_pred, y_test)
    prec_ml = precision_ml(y_pred, y_test)
    rec_ml = recall_ml(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_14_OUT_SECOND_STAGE + 'result_prob_late_fusion_stage_2.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


def metrics_cnn_median_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_preb_role_median_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_proba_role_median_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    acc_ml = accuracy_ml(y_pred, y_test)
    prec_ml = precision_ml(y_pred, y_test)
    rec_ml = recall_ml(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_14_OUT_SECOND_STAGE + 'result_median_late_fusion_stage_2.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


def metrics_cnn_max_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_pred_role_max_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "y_proba_role_max_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy_score(y_pred, y_test)
    acc_ml = accuracy_ml(y_pred, y_test)
    prec_ml = precision_ml(y_pred, y_test)
    rec_ml = recall_ml(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_14_OUT_SECOND_STAGE + 'result_max_late_fusion_stage_2.csv') as results:
        print('Hamming Loss - {.4f}'.format(hl_score), file=results)
        print('Accuracy Score - {.4f}'.format(acc_score), file=results)
        print('Accuracy Multilabel - {.4f}'.format(acc_ml), file=results)
        print('Precision Multilabel - {.4f}'.format(prec_ml), file=results)
        print('Recall Multilabel - {.4f}'.format(rec_ml), file=results)
        print('F1 Score - {.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC Score - {.4f}'.format(roc_score), file=results)
        print('Average Precision Score - {.4f}'.format(pr_score), file=results)


if __name__ == '__main__':
    metrics_cnn_sum_stage_1()
    metrics_cnn_prob_stage_1()
    metrics_cnn_median_stage_1()
    metrics_cnn_max_stage_1()
    metrics_cnn_sum_stage_2()
    metrics_cnn_prob_stage_2()
    metrics_cnn_median_stage_2()
    metrics_cnn_max_stage_2()
