import os
import sys
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
from model import cnn_svm_model
sys.path.append('src/')
from metrics import auc
from generate_structure import TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, AUDIO_MEL_SPECTROGRAM,\
 MODELS_CNN_SVM_TENSOR, MODELS_CNN_SVM_WEIGHTS_FINAL, MODELS_CNN_SVM_WEIGTHS_PER_EPOCHS, OUT_CNN_SVM_FIRST
sys.path.append('database/CAL500')
from config_cal500 import BATCH_SIZE, IMG_SIZE, TARGET_SIZE, LR, NUM_WORKERS, NUM_EPOCHS, LR_DECAY, SEED

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

model = cnn_svm_model()

plot_model(model, to_file=OUT_CNN_SVM_FIRST + 'cnn_svm_inputs_first_stage.png')

model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    lr=LR, decay=LR_DECAY), metrics=['accuracy', auc])

callbacks_list = [
    ModelCheckpoint(MODELS_CNN_SVM_WEIGTHS_PER_EPOCHS + 'weights_first_stage_{epoch:03d}.h5', save_weights_only=True, save_best_only=True),
    TensorBoard(log_dir=MODELS_CNN_SVM_TENSOR + 'first_stage/', histogram_freq=0, write_graph=True),
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25),
    EarlyStopping(monitor='val_acc', mode='max', patience=25),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, min_lr=1e-10, mode='auto', verbose=1),
    CSVLogger(OUT_CNN_SVM_FIRST + 'training.csv', append=True, separator=',')
]

STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n/valid_generator.batch_size
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=NUM_EPOCHS,
    callbacks=callbacks_list,
    verbose=1
)

model.save_weights(MODELS_CNN_SVM_WEIGHTS_FINAL + 'first_stage.h5')

score = model.evaluate_generator(valid_generator, steps=STEP_SIZE_VALID, verbose=0)

print('\nTest Loss: {:.4f} \nTest Accuracy: {:.4f}\n'.format(score[0], score[1]))

test_generator.reset()
predictions_test = model.predict_generator(test_generator,
                                           steps=STEP_SIZE_TEST,
                                           verbose=0)

train_generator.reset()
predictions_train = model.predict_generator(train_generator,
                                            steps=STEP_SIZE_TRAIN,
                                            verbose=0)

results_test = pd.DataFrame(predictions_test, columns=columns)
results_test["Song Name"] = test_generator.filenames
ordered_cols = ["Song Name"] + columns
results_test = results_test[ordered_cols]
results_test.to_csv(OUT_CNN_SVM_FIRST + "predictions_test.csv", index=False)

results_train = pd.DataFrame(predictions_train, columns=columns)
results_train["Song Name"] = train_generator.filenames
ordered_cols = ["Song Name"] + columns
results_train = results_train[ordered_cols]
results_train.to_csv(OUT_CNN_SVM_FIRST + "predictions_train.csv", index=False)


def generate_acc_graph():
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(OUT_CNN_SVM_FIRST + 'model_accuracy_first_stage.png')
    plt.close()


def generate_loss_graph():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(OUT_CNN_SVM_FIRST + 'model_loss_first_stage.png')
    plt.close()


def generate_auc_graph():
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(OUT_CNN_SVM_FIRST + 'model_auc_first_stage.png')
    plt.close()

if __name__ == '__main__':
    generate_acc_graph()
    generate_loss_graph()
    generate_auc_graph()
    K.clear_session()
