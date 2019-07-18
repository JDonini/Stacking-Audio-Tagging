import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback, CSVLogger
from keras.optimizers import RMSprop
from keras.utils import plot_model
from model import merge_model_16
sys.path.append('src')
from metrics import auc_roc, hamming_loss, ranking_loss, auc_pr, accuracy
from generate_graph import generate_acc_graph, generate_loss_graph, generate_auc_roc_graph, generate_auc_pr_graph, \
 generate_hamming_loss_graph, generate_ranking_loss_graph
from generate_structure import AUDIO_MEL_SPECTROGRAM, AUTOENCODERS_MEL_SPECTROGRAM, TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, \
 MODEL_16_TENSOR, MODEL_16_WEIGHTS_FINAL, MODEL_16_WEIGTHS_PER_EPOCHS, MODEL_16_OUT_FIRST_STAGE
sys.path.append('database')
from config_project import BATCH_SIZE, TARGET_SIZE, LR, NUM_EPOCHS, LR_DECAY, SEED

np.random.seed(SEED)
tf.set_random_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()
datagen = ImageDataGenerator(rescale=1./255)

train_generator_harmonic = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TRAIN_ANNOTATIONS),
    directory=AUDIO_MEL_SPECTROGRAM,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

test_generator_harmonic = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TEST_ANNOTATIONS),
    directory=AUDIO_MEL_SPECTROGRAM,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

validation_generator_harmonic = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(VALIDATION_ANNOTATIONS),
    directory=AUDIO_MEL_SPECTROGRAM,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

train_generator_percissive = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TRAIN_ANNOTATIONS),
    directory=AUTOENCODERS_MEL_SPECTROGRAM,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

test_generator_percissive = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TEST_ANNOTATIONS),
    directory=AUTOENCODERS_MEL_SPECTROGRAM,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

validation_generator_percissive = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(VALIDATION_ANNOTATIONS),
    directory=AUTOENCODERS_MEL_SPECTROGRAM,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)


def train_generate_multiple_input():
    while True:
        for (X1i, X2i) in zip(train_generator_harmonic, train_generator_percissive):
            yield [X1i[0], X2i[0]], X1i[1]


def test_generate_multiple_input():
    while True:
        for (X1i, X2i) in zip(test_generator_harmonic, test_generator_percissive):
            yield [X1i[0], X2i[0]], X1i[1]


def validation_generate_multiple_input():
    while True:
        for (X1i, X2i) in zip(validation_generator_harmonic, validation_generator_percissive):
            yield [X1i[0], X2i[0]], X1i[1]


STEP_SIZE_TRAIN = train_generator_harmonic.n/train_generator_harmonic.batch_size
STEP_SIZE_VALID = validation_generator_harmonic.n/validation_generator_harmonic.batch_size
STEP_SIZE_TEST = test_generator_harmonic.n/test_generator_harmonic.batch_size

if len(K.tensorflow_backend._get_available_gpus()) > 1:
    model = multi_gpu_model(merge_model_16(), gpus=len(K.tensorflow_backend._get_available_gpus()))
else:
    model = merge_model_16()

model.load_weights(MODEL_16_WEIGHTS_FINAL + 'first_stage.h5')

model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    lr=LR, decay=LR_DECAY), metrics=['accuracy', accuracy, auc_roc, auc_pr, hamming_loss, ranking_loss])

datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.datetime.now()))

callbacks_list = [
    ModelCheckpoint(MODEL_16_WEIGTHS_PER_EPOCHS + 'weights_second_stage_{epoch:03d}.h5', save_weights_only=True, save_best_only=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20),
    EarlyStopping(monitor='val_acc', mode='max', patience=20),
    TensorBoard(log_dir=MODEL_16_TENSOR + 'second_stage/' + datetime_str, histogram_freq=0, write_graph=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=12, min_lr=1e-10, mode='auto', verbose=1),
    CSVLogger(MODEL_16_OUT_SECOND_STAGE + 'training.csv', append=True, separator=',')
]

history = model.fit_generator(
    generator=train_generate_multiple_input(),
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generate_multiple_input(),
    validation_steps=STEP_SIZE_VALID,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1,
    max_queue_size=100
)

score = model.evaluate_generator(
    validation_generate_multiple_input(), steps=STEP_SIZE_VALID, max_queue_size=100)

results_testing = pd.DataFrame()
results_testing.loc[0, 'Loss'] = float('{0:.4f}'.format(score[0]))
results_testing.loc[0, 'Accuracy'] = float('{0:.4f}'.format(score[1]))
results_testing.loc[0, 'AUC-ROC'] = float('{0:.4f}'.format(score[2]))
results_testing.loc[0, 'AUC-PR'] = float('{0:.4f}'.format(score[3]))
results_testing.loc[0, 'Hamming Loss'] = float('{0:.4f}'.format(score[4]))
results_testing.loc[0, 'Ranking Loss'] = float('{0:.4f}'.format(score[5]))
results_testing.to_csv(MODEL_16_OUT_SECOND_STAGE + "testing.csv", index=False)

test_generator_harmonic.reset()
predictions = model.predict_generator(test_generate_multiple_input(),
                                      steps=STEP_SIZE_TEST,
                                      max_queue_size=100)

results = pd.DataFrame(data=(predictions > 0.5).astype(int), columns=columns)
results["song_name"] = test_generator_harmonic.filenames
ordered_cols = ["song_name"] + columns
results = results[ordered_cols]
results.to_csv(MODEL_16_OUT_SECOND_STAGE + "predictions.csv", index=False)

if __name__ == '__main__':
    K.clear_session()
    generate_acc_graph(history, MODEL_16_OUT_SECOND_STAGE, 'model_accuracy_second_stage.png')
    generate_loss_graph(history, MODEL_16_OUT_SECOND_STAGE, 'model_loss_second_stage.png')
    generate_auc_roc_graph(history, MODEL_16_OUT_SECOND_STAGE, 'model_auc_roc_second_stage.png')
    generate_auc_pr_graph(history, MODEL_16_OUT_SECOND_STAGE, 'model_auc_pr_second_stage.png')
    generate_hamming_loss_graph(history, MODEL_16_OUT_SECOND_STAGE, 'model_hamming_loss_second_stage.png')
    generate_ranking_loss_graph(history, MODEL_16_OUT_SECOND_STAGE, 'model_ranking_loss_second_stage.png')
    plot_model(model, to_file=MODEL_16_OUT_SECOND_STAGE + 'cnn_model_16_second_stage.png')
