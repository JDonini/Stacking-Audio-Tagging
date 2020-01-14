import pandas as pd
import numpy as np
import sys
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score, average_precision_score
sys.path.append('src')
from generate_structure import MODEL_12_OUT_FIRST_STAGE, MODEL_12_OUT_SECOND_STAGE, TEST_ANNOTATIONS


def accuracy(y_pred, y_true):
    return np.mean(np.equal(y_true, y_pred, dtype=np.int32))


def metrics_cnn_sum_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_pred_role_sum_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_proba_role_sum_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_FIRST_STAGE + 'result_sum_late_fusion_stage_1.csv', 'w') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


def metrics_cnn_prod_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_pred_role_prod_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_proba_role_prod_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_FIRST_STAGE + 'result_prob_late_fusion_stage_1.csv', 'w+') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


def metrics_cnn_median_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_pred_role_median_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_proba_role_median_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_FIRST_STAGE + 'result_median_late_fusion_stage_1.csv', 'w+') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


def metrics_cnn_max_stage_1():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_pred_role_max_late_fusion_stage_1.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_FIRST_STAGE + "y_proba_role_max_late_fusion_stage_1.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_FIRST_STAGE + 'result_max_late_fusion_stage_1.csv', 'w+') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


def metrics_cnn_sum_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_pred_role_sum_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_proba_role_sum_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_SECOND_STAGE + 'result_sum_late_fusion_stage_2.csv', 'w+') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


def metrics_cnn_prod_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_pred_role_prod_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_proba_role_prod_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_SECOND_STAGE + 'result_prob_late_fusion_stage_2.csv', 'w+') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


def metrics_cnn_median_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_pred_role_median_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_proba_role_median_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_SECOND_STAGE + 'result_median_late_fusion_stage_2.csv', 'w+') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


def metrics_cnn_max_stage_2():
    y_test = pd.read_csv(TEST_ANNOTATIONS, header=0, index_col=0).values
    y_pred = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_pred_role_max_late_fusion_stage_2.csv", header=0, index_col=0).values
    y_proba = pd.read_csv(MODEL_12_OUT_SECOND_STAGE + "y_proba_role_max_late_fusion_stage_2.csv", header=0, index_col=0).values

    hl_score = hamming_loss(y_pred, y_test)
    acc_score = accuracy(y_pred, y_test)
    fmeasure_score = f1_score(y_pred, y_test, average='micro')
    roc_score = roc_auc_score(y_test, y_proba, average='micro')
    pr_score = average_precision_score(y_test, y_proba, average='micro')

    with open(MODEL_12_OUT_SECOND_STAGE + 'result_max_late_fusion_stage_2.csv', 'w+') as results:
        print('Hamming Loss - {:.4f}'.format(hl_score), file=results)
        print('Accuracy - {:.4f}'.format(acc_score), file=results)
        print('F1 - {:.4f}'.format(fmeasure_score), file=results)
        print('Roc AUC - {:.4f}'.format(roc_score), file=results)
        print('Average Precision - {:.4f}'.format(pr_score), file=results)


if __name__ == '__main__':
    metrics_cnn_sum_stage_1()
    metrics_cnn_prod_stage_1()
    metrics_cnn_median_stage_1()
    metrics_cnn_max_stage_1()
    metrics_cnn_sum_stage_2()
    metrics_cnn_prod_stage_2()
    metrics_cnn_median_stage_2()
    metrics_cnn_max_stage_2()
