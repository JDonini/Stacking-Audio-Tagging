import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.optimizers import RMSprop
from keras.utils import plot_model
from keras.applications.vgg19 import VGG19
sys.path.append('src')
from generate_structure import AUDIO_MFCC, TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, \
 MODEL_13_TENSOR, MODEL_13_WEIGHTS_FINAL, MODEL_13_OUT_SECOND_STAGE
sys.path.append('config')
from config_project import BATCH_SIZE, TARGET_SIZE, LR, NUM_EPOCHS, LR_DECAY, SEED, IMG_SIZE, EARLY_STOPPING, REDUCE_LR

np.random.seed(SEED)
tf.set_random_seed(SEED)

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TRAIN_ANNOTATIONS),
    directory=AUDIO_MFCC,
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
    directory=AUDIO_MFCC,
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
    directory=AUDIO_MFCC,
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

model.load_weights(MODEL_13_WEIGHTS_FINAL + 'weights_first_stage_vgg_19.h5')

model.compile(loss='binary_crossentropy', optimizer=RMSprop(
    lr=LR, decay=LR_DECAY), metrics=['accuracy'])

datetime_str = ('{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.datetime.now()))

callbacks_list = [
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=EARLY_STOPPING),
    EarlyStopping(monitor='val_acc', mode='max', patience=EARLY_STOPPING),
    TensorBoard(log_dir=MODEL_13_TENSOR + 'second_stage/' + datetime_str, histogram_freq=0, write_graph=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=REDUCE_LR, min_lr=1e-10, mode='auto', verbose=1),
    CSVLogger(MODEL_13_OUT_SECOND_STAGE + 'training_vgg_19.csv', append=True, separator=',')
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

test_generator.reset()
predictions = model.predict_generator(test_generator,
                                      steps=STEP_SIZE_TEST,
                                      max_queue_size=100)

test_generator.reset()
results = pd.DataFrame(data=predictions, columns=columns)
results["song_name"] = test_generator.filenames
ordered_cols = ["song_name"] + columns
results = results[ordered_cols]
results.to_csv(MODEL_13_OUT_SECOND_STAGE + "predictions_vgg_19.csv", index=False)


if __name__ == '__main__':
    k.clear_session()
    plot_model(model, to_file=MODEL_13_OUT_SECOND_STAGE + 'cnn_model_stage_2_vgg_19.png')
