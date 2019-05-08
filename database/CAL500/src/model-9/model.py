import sys
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
sys.path.append('database/CAL500')
from config_cal500 import IMG_SIZE


def cnn_cnn_autoencoders_model():
    # Encoder
    input_img = Input(shape=(IMG_SIZE))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    encoded = MaxPooling2D()(x)

    # Classification
    classification = Flatten()(encoded)
    classification = Dense(128, activation='relu')(classification)
    classification = Dense(units=97, activation='sigmoid', name='classification')(classification)

    # Decoder
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    y = UpSampling2D()(y)
    y = Dropout(rate=0.2)(y)
    y = Conv2D(128, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D()(y)

    y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D()(y)
    y = Dropout(rate=0.4)(y)
    y = Conv2D(64, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D()(y)
    y = Dropout(rate=0.3)(y)

    y = Conv2D(32, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D()(y)
    y = Dropout(rate=0.4)(y)
    y = Conv2D(32, (3, 3), activation='relu', padding='same')(y)
    y = UpSampling2D()(y)
    y = Dropout(rate=0.3)(y)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='autoencoder')(y)

    return Model(inputs=input_img, outputs=[classification, decoded])
