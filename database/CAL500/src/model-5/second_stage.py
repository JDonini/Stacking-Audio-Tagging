import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback, CSVLogger
from keras.optimizers import RMSprop
from keras.models import load_model, Model
from model import bottleneck_features
sys.path.append('src')
from metrics import auc_roc, auc_pr, hamming_loss, ranking_loss
from generate_structure import TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, AUDIO_MEL_SPECTROGRAM, \
 MODEL_5_TENSOR, MODEL_5_WEIGHTS_FINAL, MODEL_5_WEIGTHS_PER_EPOCHS, MODEL_5_OUT_FIRST_STAGE, MODEL_5_OUT_SECOND_STAGE
sys.path.append('database/CAL500')
from config_cal500 import BATCH_SIZE, TARGET_SIZE, LR, NUM_EPOCHS, LR_DECAY, SEED

np.random.seed(SEED)
tf.set_random_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()

datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_dataframe(
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

test_generator = datagen.flow_from_dataframe(
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

valid_generator = datagen.flow_from_dataframe(
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

STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n/valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size

train_data = np.load(open(MODEL_5_OUT_FIRST_STAGE + 'features_train.npy', 'rb'))
train_labels = np.array([0] * (train_generator.n // 2) + [1] * (train_generator.n // 2))

validation_data = np.load(open(MODEL_5_OUT_FIRST_STAGE + 'features_validation.npy', 'rb'))
validation_labels = np.array([0] * (valid_generator.n // 2) + [1] * (valid_generator.n // 2))

model = bottleneck_features()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    lr=LR, decay=LR_DECAY), metrics=['accuracy', auc_roc, auc_pr, hamming_loss, ranking_loss])

datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.datetime.now()))

callbacks_list = [
    ModelCheckpoint(MODEL_5_WEIGTHS_PER_EPOCHS + 'weights_second_stage_{epoch:03d}.h5', save_weights_only=True, save_best_only=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20),
    EarlyStopping(monitor='val_acc', mode='max', patience=20),
    TensorBoard(log_dir=MODEL_5_TENSOR + 'second_stage/' + datetime_str, histogram_freq=0, write_graph=False, write_images=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=12, min_lr=1e-10, mode='auto', verbose=1),
    CSVLogger(MODEL_5_OUT_SECOND_STAGE + 'training.csv', append=True, separator=',')
]

# history = model.fit_generator(
#     generator=train_generator,
#     steps_per_epoch=STEP_SIZE_TRAIN,
#     validation_data=valid_generator,
#     validation_steps=STEP_SIZE_VALID,
#     epochs=NUM_EPOCHS,
#     callbacks=callbacks_list,
#     verbose=1,
#     max_queue_size=100
# )

history = model.fit(train_data, train_labels,
                    batch_size=2,
                    epochs=NUM_EPOCHS,
                    validation_data=(validation_data, validation_labels)
                    )

if __name__ == '__main__':
    K.clear_session()
    plot_model(model, to_file=MODEL_5_OUT_SECOND_STAGE + 'cnn_model_5_second_stage.png')
