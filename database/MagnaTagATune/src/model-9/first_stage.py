import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import RMSprop
from model import merge_model_9
sys.path.append('src')
from generate_structure import AUDIO_STFT_HARMONIC, AUDIO_STFT_PERCUSSIVE, TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, \
 VALIDATION_ANNOTATIONS, MODEL_9_TENSOR, MODEL_9_WEIGHTS_FINAL, MODEL_9_OUT_FIRST_STAGE
sys.path.append('config')
from config_project import BATCH_SIZE, TARGET_SIZE, LR, NUM_EPOCHS, LR_DECAY, SEED, EARLY_STOPPING, REDUCE_LR

np.random.seed(SEED)
tf.set_random_seed(SEED)

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()
datagen = ImageDataGenerator(rescale=1./255)

train_generator_harmonic = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TRAIN_ANNOTATIONS),
    directory=AUDIO_STFT_HARMONIC,
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
    directory=AUDIO_STFT_HARMONIC,
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
    directory=AUDIO_STFT_HARMONIC,
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
    directory=AUDIO_STFT_PERCUSSIVE,
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
    directory=AUDIO_STFT_PERCUSSIVE,
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
    directory=AUDIO_STFT_PERCUSSIVE,
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

model = merge_model_9()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    lr=LR, decay=LR_DECAY), metrics=['accuracy'])

datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.datetime.now()))

callbacks_list = [
    ModelCheckpoint(MODEL_9_WEIGHTS_FINAL + 'weights_first_stage.h5', save_weights_only=True, save_best_only=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOPPING),
    EarlyStopping(monitor='val_acc', mode='max', patience=EARLY_STOPPING),
    TensorBoard(log_dir=MODEL_9_TENSOR + 'first_stage/' + datetime_str, histogram_freq=0, write_graph=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=REDUCE_LR, min_lr=1e-10, mode='auto', verbose=1),
    CSVLogger(MODEL_9_OUT_FIRST_STAGE + 'training.csv', append=True, separator=',')
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

test_generator_harmonic.reset()
predictions = model.predict_generator(test_generate_multiple_input(),
                                      steps=STEP_SIZE_TEST,
                                      max_queue_size=100)

results_proba = pd.DataFrame(data=predictions, columns=columns)
results_proba["song_name"] = test_generator_harmonic.filenames
ordered_cols = ["song_name"] + columns
results_proba = results_proba[ordered_cols]
results_proba.to_csv(MODEL_9_OUT_FIRST_STAGE + "y_proba_stage_1.csv", index=False)

results_pred = pd.DataFrame(data=(predictions > 0.5).astype(int), columns=columns)
results_pred["song_name"] = test_generator_harmonic.filenames
ordered_cols = ["song_name"] + columns
results_pred = results_pred[ordered_cols]
results_pred.to_csv(MODEL_9_OUT_FIRST_STAGE + "y_pred_stage_1.csv", index=False)

if __name__ == '__main__':
    k.clear_session()
    plot_model(model, to_file=MODEL_9_OUT_FIRST_STAGE + 'cnn_model_stage_1.png')
