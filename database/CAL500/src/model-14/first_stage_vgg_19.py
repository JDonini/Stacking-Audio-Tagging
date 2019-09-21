import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.applications.vgg19 import VGG19
sys.path.append('src')
from metrics import auc_roc, hamming_loss, ranking_loss, auc_pr
from generate_graph import generate_acc_graph, generate_loss_graph, generate_auc_roc_graph, generate_auc_pr_graph, \
 generate_hamming_loss_graph, generate_ranking_loss_graph
from generate_structure import AUDIO_STFT, TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, \
 MODEL_14_TENSOR, MODEL_14_WEIGHTS_FINAL, MODEL_14_OUT_FIRST_STAGE
sys.path.append('database')
from config_project import BATCH_SIZE, TARGET_SIZE, LR, NUM_EPOCHS, LR_DECAY, SEED, IMG_SIZE

np.random.seed(SEED)
tf.set_random_seed(SEED)

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()
datagen = ImageDataGenerator(rescale=1./255)

train_generator= datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TRAIN_ANNOTATIONS),
    directory=AUDIO_STFT,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

test_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TEST_ANNOTATIONS),
    directory=AUDIO_STFT,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(VALIDATION_ANNOTATIONS),
    directory=AUDIO_STFT,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='other',
    target_size=TARGET_SIZE
)

STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size
STEP_SIZE_VALID = validation_generator.n/validation_generator.batch_size

model = Sequential()
model.add(VGG19(weights=None, include_top=False, input_tensor=Input(shape=IMG_SIZE)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(97, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    lr=LR, decay=LR_DECAY), metrics=['accuracy', auc_roc, auc_pr, hamming_loss, ranking_loss])

datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.datetime.now()))

callbacks_list = [
    ModelCheckpoint(MODEL_14_WEIGHTS_FINAL + 'weights_first_stage_vgg_19.h5', save_weights_only=True, save_best_only=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20),
    EarlyStopping(monitor='val_acc', mode='max', patience=20),
    TensorBoard(log_dir=MODEL_14_TENSOR + 'first_stage/' + datetime_str, histogram_freq=0, write_graph=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=12, min_lr=1e-10, mode='auto', verbose=1),
    CSVLogger(MODEL_14_OUT_FIRST_STAGE + 'training_vgg_19.csv', append=True, separator=',')
]

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1,
    max_queue_size=100
)

score = model.evaluate_generator(
    validation_generator, steps=STEP_SIZE_VALID, max_queue_size=100)

results_testing = pd.DataFrame()
results_testing.loc[0, 'Loss'] = float('{0:.4f}'.format(score[0]))
results_testing.loc[0, 'Accuracy'] = float('{0:.4f}'.format(score[1]))
results_testing.loc[0, 'AUC-ROC'] = float('{0:.4f}'.format(score[2]))
results_testing.loc[0, 'AUC-PR'] = float('{0:.4f}'.format(score[3]))
results_testing.loc[0, 'Hamming Loss'] = float('{0:.4f}'.format(score[4]))
results_testing.loc[0, 'Ranking Loss'] = float('{0:.4f}'.format(score[5]))
results_testing.to_csv(MODEL_14_OUT_FIRST_STAGE + "testing_vgg_19.csv", index=False)

predictions = model.predict_generator(generator=test_generator,
                                      steps=STEP_SIZE_TEST,
                                      max_queue_size=100)

test_generator.reset()
results = pd.DataFrame(data=predictions, columns=columns)
results["song_name"] = test_generator.filenames
ordered_cols = ["song_name"] + columns
results = results[ordered_cols]
results.to_csv(MODEL_14_OUT_FIRST_STAGE + "predictions_vgg_19.csv", index=False)


if __name__ == '__main__':
    k.clear_session()
    generate_acc_graph(history, MODEL_14_OUT_FIRST_STAGE, 'model_accuracy_first_stage_vgg_19.png')
    generate_loss_graph(history, MODEL_14_OUT_FIRST_STAGE, 'model_loss_first_stage_vgg_19.png')
    generate_auc_roc_graph(history, MODEL_14_OUT_FIRST_STAGE, 'model_auc_roc_first_stage_vgg_19.png')
    generate_auc_pr_graph(history, MODEL_14_OUT_FIRST_STAGE, 'model_auc_pr_first_stage_vgg_19.png')
    generate_hamming_loss_graph(history, MODEL_14_OUT_FIRST_STAGE, 'model_hamming_loss_first_stage_vgg_19.png')
    generate_ranking_loss_graph(history, MODEL_14_OUT_FIRST_STAGE, 'model_ranking_loss_second_stage_vgg_19.png')
    plot_model(model, to_file=MODEL_14_OUT_FIRST_STAGE + 'cnn_model_first_stage_vgg_19.png')
