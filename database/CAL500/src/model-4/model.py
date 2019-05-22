import sys
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import backend as K
sys.path.append('database/CAL500')
from config_cal500 import IMG_SIZE


def cnn_cnn_model_4_arq_1():
    input_1 = Input(shape=(IMG_SIZE))
    x = Conv2D(32, (3, 3), activation='relu')(input_1)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.25)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.2)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    return input_1, x


def cnn_cnn_model_4_arq_2():
    input_2 = Input(shape=(IMG_SIZE))
    y = Conv2D(32, (3, 3), activation='relu')(input_2)
    y = BatchNormalization()(y)
    y = Dropout(rate=0.2)(y)
    y = MaxPooling2D()(y)

    y = Conv2D(32, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(rate=0.35)(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(rate=0.35)(y)
    y = MaxPooling2D()(y)

    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(rate=0.3)(y)
    y = Conv2D(128, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(rate=0.4)(y)
    y = MaxPooling2D()(y)

    y = Conv2D(128, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(rate=0.3)(y)
    y = Conv2D(256, (3, 3), activation='relu')(y)
    y = BatchNormalization()(y)
    y = Dropout(rate=0.4)(y)
    y = MaxPooling2D()(y)
    y = Flatten()(y)

    return input_2, y


def merge_model_4():
    input_arq_1, model_arq_1 = cnn_cnn_model_4_arq_1()
    input_arq_2, model_arq_2 = cnn_cnn_model_4_arq_2()

    merge = concatenate([model_arq_1, model_arq_2])

    hidden_1 = Dense(512, activation='relu')(merge)
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    output = Dense(97, activation='sigmoid')(hidden_2)

    return Model(inputs=[input_arq_1, input_arq_2], outputs=output)
