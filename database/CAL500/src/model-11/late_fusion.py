import os
import sys
import numpy as np
import pandas as pd
sys.path.append('src')
from metrics import auc_roc, hamming_loss, ranking_loss, auc_pr
from generate_structure import MODEL_11_OUT_FIRST_STAGE, MODEL_11_OUT_SECOND_STAGE

columns = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_vgg_19.csv").columns[1:].tolist()

predict_vgg_19_stage_1 = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_vgg_19.csv", usecols=columns)
predict_inception_resnet_v2_stage_1 = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_xception.csv", usecols=columns)
predict_inception_v3_stage_1 = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_inception_v3.csv", usecols=columns)

predict_vgg_19_stage_2 = pd.read_csv(MODEL_11_OUT_SECOND_STAGE + "predictions_vgg_19.csv", usecols=columns)
predict_inception_resnet_v2_stage_2 = pd.read_csv(MODEL_11_OUT_SECOND_STAGE + "predictions_xception.csv", usecols=columns)
predict_inception_v3_stage_2 = pd.read_csv(MODEL_11_OUT_SECOND_STAGE + "predictions_inception_v3.csv", usecols=columns)


def late_fusion_stage_1():
    all_prediction_stage_1 = np.array([predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1,
                                      predict_inception_v3_stage_1])

    predict_stage_1_sum = all_prediction_stage_1.sum(0).argmax(1)
    predict_stage_1_prod = all_prediction_stage_1.prod(0).argmax(1)
    predict_stage_1_median = np.median(all_prediction_stage_1, 0).argmax(1)
    predict_stage_1_max = np.max(all_prediction_stage_1, 0).argmax(1)


def late_fusion_stage_2():
    all_prediction_stage_2 = np.array(predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2,
                                      predict_inception_v3_stage_2)

    predict_stage_2_sum = all_prediction_stage_2.sum(0).argmax(1)
    predict_stage_2_prod = all_prediction_stage_2.prod(0).argmax(1)
    predict_stage_2_median = np.median(all_prediction_stage_2, 0).argmax(1)
    predict_stage_2_max = np.max(all_prediction_stage_2, 0).argmax(1)


if __name__ == '__main__':
    late_fusion_stage_1()
    late_fusion_stage_2()
