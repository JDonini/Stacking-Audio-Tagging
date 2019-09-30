import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
sys.path.append('src')
from metrics import auc_roc, hamming_loss, ranking_loss, auc_pr
from generate_structure import MODEL_11_OUT_FIRST_STAGE, MODEL_11_OUT_SECOND_STAGE

sys.path.append('config')
from config_project import BATCH_SIZE, TARGET_SIZE, LR, NUM_EPOCHS, LR_DECAY, SEED, IMG_SIZE, EARLY_STOPPING, REDUCE_LR

np.random.seed(SEED)
tf.set_random_seed(SEED)

columns = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_vgg_19.csv").columns[:].tolist()

predict_vgg_19_stage_1 = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_vgg_19.csv", usecols=columns)
predict_vgg_19_stage_1 = predict_vgg_19_stage_1.set_index('song_name')

predict_inception_resnet_v2_stage_1 = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_xception.csv", usecols=columns)
predict_inception_resnet_v2_stage_1 = predict_inception_resnet_v2_stage_1.set_index('song_name')

predict_inception_v3_stage_1 = pd.read_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_inception_v3.csv", usecols=columns)
predict_inception_v3_stage_1 = predict_inception_v3_stage_1.set_index('song_name')

predict_vgg_19_stage_2 = pd.read_csv(MODEL_11_OUT_SECOND_STAGE + "predictions_vgg_19.csv", usecols=columns)
predict_vgg_19_stage_2 = predict_vgg_19_stage_2.set_index('song_name')

predict_inception_resnet_v2_stage_2 = pd.read_csv(MODEL_11_OUT_SECOND_STAGE + "predictions_xception.csv", usecols=columns)
predict_inception_resnet_v2_stage_2 = predict_inception_resnet_v2_stage_2.set_index('song_name')

predict_inception_v3_stage_2 = pd.read_csv(MODEL_11_OUT_SECOND_STAGE + "predictions_inception_v3.csv", usecols=columns)
predict_inception_v3_stage_2 = predict_inception_v3_stage_2.set_index('song_name')


def late_fusion_stage_1():
    predict_stage_1_sum = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).sum(level=0)
    predict_stage_1_sum = (predict_stage_1_sum > 0.5).astype(int)

    predict_stage_1_prod = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).prod(level=0)
    predict_stage_1_prod = (predict_stage_1_prod > 0.5).astype(int)

    predict_stage_1_median = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).median(level=0)
    predict_stage_1_median = (predict_stage_1_median > 0.5).astype(int)

    predict_stage_1_max = pd.concat(
        [predict_vgg_19_stage_1, predict_inception_resnet_v2_stage_1, predict_inception_v3_stage_1]).max(level=0)
    predict_stage_1_max = (predict_stage_1_max > 0.5).astype(int)


def late_fusion_stage_2():
    predict_stage_2_sum = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).sum(level=0)
    predict_stage_2_sum = (predict_stage_2_sum > 0.5).astype(int)

    predict_stage_2_prod = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).prod(level=0)
    predict_stage_2_prod = (predict_stage_2_prod > 0.5).astype(int)

    predict_stage_2_median = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).median(level=0)
    predict_stage_2_median = (predict_stage_2_median > 0.5).astype(int)

    predict_stage_2_max = pd.concat(
        [predict_vgg_19_stage_2, predict_inception_resnet_v2_stage_2, predict_inception_v3_stage_2]).max(level=0)
    predict_stage_2_max = (predict_stage_2_max > 0.5).astype(int)

STEP_SIZE_TEST = test_generator.n/test_generator.batch_size

test_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TEST_ANNOTATIONS),
    directory=AUDIO_CHROMAGRAM,
    x_col='song_name',
    y_col=columns[1:],
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

score = model.evaluate_generator(
    test_generator, steps=STEP_SIZE_VALID, max_queue_size=100)

results_testing = pd.DataFrame()
results_testing.loc[0, 'Loss'] = float('{0:.4f}'.format(score[0]))
results_testing.loc[0, 'Accuracy'] = float('{0:.4f}'.format(score[1]))
results_testing.loc[0, 'AUC-ROC'] = float('{0:.4f}'.format(score[2]))
results_testing.loc[0, 'AUC-PR'] = float('{0:.4f}'.format(score[3]))
results_testing.loc[0, 'Hamming Loss'] = float('{0:.4f}'.format(score[4]))
results_testing.loc[0, 'Ranking Loss'] = float('{0:.4f}'.format(score[5]))
results_testing.to_csv(MODEL_11_OUT_FIRST_STAGE + "testing_inception_v3.csv", index=False)

predictions = model.predict_generator(generator=test_generator,
                                      steps=STEP_SIZE_TEST,
                                      max_queue_size=100)

test_generator.reset()
results = pd.DataFrame(data=predictions, columns=columns)
results["song_name"] = test_generator.filenames
ordered_cols = ["song_name"] + columns
results = results[ordered_cols]
results.to_csv(MODEL_11_OUT_FIRST_STAGE + "predictions_inception_v3.csv", index=False)


if __name__ == '__main__':
    late_fusion_stage_1()
    late_fusion_stage_2()
    k.clear_session()
    generate_acc_graph(history, MODEL_11_OUT_FIRST_STAGE, 'model_accuracy_first_stage_inception_v3.png')
    generate_loss_graph(history, MODEL_11_OUT_FIRST_STAGE, 'model_loss_first_stage_inception_v3.png')
    generate_auc_roc_graph(history, MODEL_11_OUT_FIRST_STAGE, 'model_auc_roc_first_stage_inception_v3.png')
    generate_auc_pr_graph(history, MODEL_11_OUT_FIRST_STAGE, 'model_auc_pr_first_stage_inception_v3.png')
    generate_hamming_loss_graph(history, MODEL_11_OUT_FIRST_STAGE, 'model_hamming_loss_first_stage_inception_v3.png')
    generate_ranking_loss_graph(history, MODEL_11_OUT_FIRST_STAGE, 'model_ranking_loss_second_stage_inception_v3.png')
    plot_model(model, to_file=MODEL_11_OUT_FIRST_STAGE + 'cnn_model_first_stage_inception_v3.png')