import sys
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
sys.path.append('database/CAL500')
from config_cal500 import IMG_SIZE


def cnn_cnn_autoencoders_model():
    # Encoder
    input_img = Input(shape=(IMG_SIZE))
    x = Conv2D(16, (3, 3), padding='same')(input_img)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(30)(x)
    x = Activation('relu')

    x = Conv2D(8, (3, 3), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)

    return Model(inputs=input_img, outputs=x)
