import sys
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
sys.path.append('database/CAL500')
from config_cal500 import IMG_SIZE


def cnn_svm_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), data_format='channels_last', input_shape=(IMG_SIZE), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(32, (3, 3), data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(64, (3, 3), data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3, 3), data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    model.add(Conv2D(128, (3, 3), data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3, 3), data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(256, (3, 3), data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))

    model.add(Dense(units=97, activation='sigmoid'))

    return model
