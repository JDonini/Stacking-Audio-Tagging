import sys
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
sys.path.append('src')
from generate_structure import TRAIN_ANNOTATIONS, VALIDATION_ANNOTATIONS, AUDIO_MEL_SPECTROGRAM, MODEL_AUTOENCODERS
sys.path.append('config')
from config_project import SEED, BATCH_SIZE, TARGET_SIZE, NUM_EPOCHS, IMG_SIZE

np.random.seed(SEED)
tf.set_random_seed(SEED)

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TRAIN_ANNOTATIONS),
    directory=AUDIO_MEL_SPECTROGRAM,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='input',
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
    class_mode='input',
    target_size=TARGET_SIZE
)


def autoencoders():
    input_img = Input(shape=IMG_SIZE)

    encoded = Conv2D(128, (3, 3), padding='same')(input_img)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(64, (3, 3), padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(32, (3, 3), padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    encoded = Conv2D(16, (3, 3), padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    decoded = Conv2D(16, (3, 3), padding='same')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(32, (3, 3), padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(64, (3, 3), padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(128, (3, 3), padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

    decoded = Conv2D(3, (3, 3), padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('sigmoid')(decoded)

    return Model(input_img, decoded)


STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n/valid_generator.batch_size

model = autoencoders()

model.compile(optimizer='adam', loss='mean_squared_error')

callbacks_list = [
    EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=12, min_lr=1e-10, mode='auto', verbose=1),
]

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=NUM_EPOCHS,
    verbose=1,
    callbacks=callbacks_list
)

model.save(MODEL_AUTOENCODERS + 'model_mel_spectrogram.h5')
print("Saved model to disk")
