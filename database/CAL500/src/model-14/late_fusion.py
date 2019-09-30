import os
import sys
import numpy as np
import pandas as pd
sys.path.append('src')
from metrics import auc_roc, hamming_loss, ranking_loss, auc_pr
from generate_structure import MODEL_14_OUT_FIRST_STAGE, MODEL_14_OUT_SECOND_STAGE

columns = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "predictions_vgg_19.csv").columns[:].tolist()

predict_vgg_19_stage_1 = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "predictions_vgg_19.csv", usecols=columns)
predict_vgg_19_stage_1 = predict_vgg_19_stage_1.set_index('song_name')

predict_inception_resnet_v2_stage_1 = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "predictions_xception.csv", usecols=columns)
predict_inception_resnet_v2_stage_1 = predict_inception_resnet_v2_stage_1.set_index('song_name')

predict_inception_v3_stage_1 = pd.read_csv(MODEL_14_OUT_FIRST_STAGE + "predictions_inception_v3.csv", usecols=columns)
predict_inception_v3_stage_1 = predict_inception_v3_stage_1.set_index('song_name')

predict_vgg_19_stage_2 = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "predictions_vgg_19.csv", usecols=columns)
predict_vgg_19_stage_2 = predict_vgg_19_stage_2.set_index('song_name')

predict_inception_resnet_v2_stage_2 = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "predictions_xception.csv", usecols=columns)
predict_inception_resnet_v2_stage_2 = predict_inception_resnet_v2_stage_2.set_index('song_name')

predict_inception_v3_stage_2 = pd.read_csv(MODEL_14_OUT_SECOND_STAGE + "predictions_inception_v3.csv", usecols=columns)
predict_inception_v3_stage_2 = predict_inception_v3_stage_2.set_index('song_name')


def late_fusion_stage_1():
    predict_stage_1_sum = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).sum(level=0)
    predict_stage_1_prod = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).prod(level=0)
    predict_stage_1_median = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).median(level=0)
    predict_stage_1_max = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).max(level=0)


def late_fusion_stage_2():
    predict_stage_2_sum = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).sum(level=0)
    predict_stage_2_prod = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).prod(level=0)
    predict_stage_2_median = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).median(level=0)
    predict_stage_2_max = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).max(level=0)


if __name__ == '__main__':
    late_fusion_stage_1()
    late_fusion_stage_2()
