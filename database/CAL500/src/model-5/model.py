import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras import backend as K
sys.path.append('src')
from generate_structure import MODEL_5_OUT_FIRST_STAGE
sys.path.append('database/CAL500')
from config_cal500 import IMG_SIZE


def cnn_cnn_model_5():
    input = Input(shape=(IMG_SIZE))

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

    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    hidden_1 = Dense(512, activation='relu')(x)
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    output = Dense(97, activation='sigmoid')(hidden_2)

    return Model(inputs=input, outputs=output)


def image_model_5():
    input = Input(shape=(IMG_SIZE))

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

    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.3)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)

    return input, x


def vector_model_5():
    features = pd.read_csv(MODEL_5_OUT_FIRST_STAGE + "features.csv", delimiter=',', index_col=None)
    print(features)
    return features


def merge_model_5():
    input_arq_1, model_arq_1 = image_model_5()
    input_arq_2, model_arq_2 = vector_model_5()

    merge = concatenate([model_arq_1, model_arq_2])

    hidden_1 = Dense(512, activation='relu')(merge)
    hidden_2 = Dense(256, activation='relu')(hidden_1)
    output = Dense(97, activation='sigmoid')(hidden_2)

    return Model(inputs=[input_arq_1, input_arq_2], outputs=output)
