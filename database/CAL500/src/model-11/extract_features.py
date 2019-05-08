import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, Callback, CSVLogger
from keras.optimizers import RMSprop
from model import cnn_svm_svm_model
sys.path.append('src/')
from metrics import auc
from generate_structure import TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, AUDIO_MEL_SPECTROGRAM,\
 MODELS_CNN_SVM_SVM_TENSOR, MODELS_CNN_SVM_SVM_WEIGHTS_FINAL, MODELS_CNN_SVM_SVM_WEIGTHS_PER_EPOCHS, OUT_CNN_SVM_SVM_FIRST
sys.path.append('database/CAL500')
from config_cal500 import BATCH_SIZE, IMG_SIZE, TARGET_SIZE, LR, NUM_WORKERS, NUM_EPOCHS, LR_DECAY, SEED

np.random.seed(SEED)
tf.set_random_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()

datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_dataframe(
    dataframe=pd.concat(
        map(pd.read_csv, [TRAIN_ANNOTATIONS, VALIDATION_ANNOTATIONS])),
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

model = cnn_svm_svm_model()

plot_model(model, to_file=OUT_CNN_SVM_SVM_FIRST + 'cnn_svm_inputs_extract_features.png')

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=LR, decay=LR_DECAY))

STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size

train_generator.reset()
predictions_train = model.predict_generator(train_generator,
                                            steps=STEP_SIZE_TRAIN,
                                            verbose=1)

results_train = pd.DataFrame(predictions_train, columns=columns)
results_train["Song Name"] = train_generator.filenames
ordered_cols = ["Song Name"] + columns
results_train = results_train[ordered_cols]
results_train.to_csv(OUT_CNN_SVM_SVM_FIRST + "predictions_train.csv", index=False)

test_generator.reset()
predictions_test = model.predict_generator(test_generator,
                                           steps=STEP_SIZE_TEST,
                                           verbose=1)

results_test = pd.DataFrame(predictions_test, columns=columns)
results_test["Song Name"] = test_generator.filenames
ordered_cols = ["Song Name"] + columns
results_test = results_test[ordered_cols]
results_test.to_csv(OUT_CNN_SVM_SVM_FIRST + "predictions_test.csv", index=False)

if __name__ == '__main__':
    K.clear_session()
