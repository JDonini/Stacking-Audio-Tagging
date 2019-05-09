import os
import sys
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback, CSVLogger
from keras.optimizers import RMSprop, Adam, SGD
from model import cnn_cnn_model_1
sys.path.append('src')
from metrics import auc_roc, hamming_loss
from generate_structure import TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, AUDIO_MEL_SPECTROGRAM, \
 MODEL_1_TENSOR, MODEL_1_WEIGHTS_FINAL, MODEL_1_WEIGTHS_PER_EPOCHS, MODEL_1_OUT_FIRST_STAGE
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
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size
STEP_SIZE_VALID = valid_generator.n/valid_generator.batch_size

model = cnn_cnn_model_1()

model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    lr=LR, decay=LR_DECAY), metrics=[hamming_loss])

datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.datetime.now()))

callbacks_list = [
    ModelCheckpoint(MODEL_1_WEIGTHS_PER_EPOCHS + 'weights_first_stage_{epoch:03d}.h5', save_weights_only=True, save_best_only=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20),
    EarlyStopping(monitor='val_acc', mode='max', patience=20),
    TensorBoard(log_dir=MODEL_1_TENSOR + 'first_stage/' + datetime_str, histogram_freq=0, write_graph=False, write_images=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-10, mode='auto', verbose=1),
    CSVLogger(MODEL_1_OUT_FIRST_STAGE + 'training.csv', append=True, separator=',')
]

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1,
    max_queue_size=100
)

model.save_weights(MODEL_1_WEIGHTS_FINAL + 'first_stage.h5')

score = model.evaluate_generator(
    valid_generator, steps=STEP_SIZE_VALID, verbose=0, max_queue_size=100)

results_testing = pd.DataFrame(columns=["Test Loss", "Test Acc"])
results_testing.loc[0, 'Test Loss'] = float("{0:.4f}".format(score[0]))
results_testing.loc[0, 'Test Acc'] = float("{0:.4f}".format(score[1]*100))
results_testing.loc[0, 'Auc'] = float("{0:.4f}".format(score[2]*100))
results_testing.to_csv(MODEL_1_OUT_FIRST_STAGE + "testing.csv", index=False)

test_generator.reset()
predictions = model.predict_generator(test_generator,
                                      steps=STEP_SIZE_TEST,
                                      verbose=0,
                                      max_queue_size=100)

predictions_bool = (predictions > 0.5)
predictions = predictions_bool.astype(int)

results = pd.DataFrame(predictions, columns=columns)
results["song_name"] = test_generator.filenames
ordered_cols = ["song_name"] + columns
results = results[ordered_cols]
results.to_csv(MODEL_1_OUT_FIRST_STAGE + "predictions.csv", index=False)


def generate_acc_graph():
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(MODEL_1_OUT_FIRST_STAGE + 'model_accuracy_first_stage.png')
    plt.close()


def generate_loss_graph():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(MODEL_1_OUT_FIRST_STAGE + 'model_loss_first_stage.png')
    plt.close()


def generate_auc_graph():
    plt.plot(history.history['auc_roc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(MODEL_1_OUT_FIRST_STAGE + 'model_auc_roc_first_stage.png')
    plt.close()


if __name__ == '__main__':
    K.clear_session()
    generate_acc_graph()
    generate_loss_graph()
    generate_auc_graph()
    plot_model(model, to_file=MODEL_1_OUT_FIRST_STAGE + 'cnn_model_1_first_stage.png')
