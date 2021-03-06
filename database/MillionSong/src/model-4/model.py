import sys
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
sys.path.append('config')
from config_project import IMG_SIZE, IMG_SIZE_AUTOENCODERS


def cnn_cnn_model_4_s1():
    input = Input(shape=IMG_SIZE)

    x = Conv2D(16, (3, 3), activation='relu')(input)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    hidden_1 = Dense(256, activation='relu')(x)
    hidden_2 = Dense(128, activation='relu')(hidden_1)
    output = Dense(265, activation='sigmoid')(hidden_2)

    return Model(inputs=input, outputs=output)


def cnn_cnn_model_4_s2():
    input = Input(shape=IMG_SIZE_AUTOENCODERS)

    x = Conv2D(16, (3, 3), activation='relu')(input)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    hidden_1 = Dense(256, activation='relu')(x)
    hidden_2 = Dense(128, activation='relu')(hidden_1)
    output = Dense(265, activation='sigmoid')(hidden_2)

    return Model(inputs=input, outputs=output)
