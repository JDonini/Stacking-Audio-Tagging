import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pylab
from keras.models import Model
from keras.layers import Input, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from keras.utils.training_utils import multi_gpu_model
from keras import backend as k
sys.path.append('src')
from generate_structure import TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS, AUDIO_STFT_PERCUSSIVE, \
 AUTOENCODERS_STFT_PERCUSSIVE
sys.path.append('database')
from config_project import SEED, BATCH_SIZE, TARGET_SIZE, NUM_EPOCHS, IMG_SIZE, FIG_SIZE_AUTOENCODERS

np.random.seed(SEED)
tf.set_random_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TRAIN_ANNOTATIONS),
    directory=AUDIO_STFT_PERCUSSIVE,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='input',
    target_size=TARGET_SIZE
)

test_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(TEST_ANNOTATIONS),
    directory=AUDIO_STFT_PERCUSSIVE,
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
    directory=AUDIO_STFT_PERCUSSIVE,
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

    encoded = Conv2D(64, (3, 3), padding='same')(input_img)
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
    encoded = Conv2D(8, (3, 3), padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Activation('relu')(encoded)

    encoded = MaxPooling2D((2, 2), padding='same')(encoded)

    decoded = Conv2D(8, (3, 3), padding='same')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Activation('relu')(decoded)
    decoded = UpSampling2D((2, 2))(decoded)

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

    decoded = Conv2D(3, (3, 3), padding='same')(decoded)
    decoded = BatchNormalization()(decoded)

    decoded = Activation('sigmoid')(decoded)

    return Model(input_img, decoded)


STEP_SIZE_TRAIN = train_generator.n/train_generator.batch_size
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size
STEP_SIZE_VALID = valid_generator.n/valid_generator.batch_size

if len(k.tensorflow_backend._get_available_gpus()) > 1:
    model = multi_gpu_model(autoencoders(), gpus=len(k.tensorflow_backend._get_available_gpus()))
else:
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
    callbacks=callbacks_list,
    max_queue_size=100
)

score = model.evaluate_generator(
    valid_generator, steps=STEP_SIZE_VALID, max_queue_size=100)

train_generator.reset()
restored_train_imgs = model.predict_generator(train_generator,
                                              steps=STEP_SIZE_TRAIN,
                                              max_queue_size=100)

test_generator.reset()
restored_test_imgs = model.predict_generator(test_generator,
                                             steps=STEP_SIZE_TEST,
                                             max_queue_size=100)

valid_generator.reset()
restored_valid_imgs = model.predict_generator(valid_generator,
                                              steps=STEP_SIZE_VALID,
                                              max_queue_size=100)


def generate_train_autoencoders():
    for audio_name in train_generator.filenames:
        print('Generate Autoencoders : {}'.format(audio_name))
        pylab.figure(figsize=FIG_SIZE_AUTOENCODERS)
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        pylab.imshow(restored_train_imgs[train_generator.filenames.index(audio_name)])
        pylab.savefig(AUTOENCODERS_STFT_PERCUSSIVE + audio_name, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        pylab.close()


def generate_test_autoencoders():
    for audio_name in test_generator.filenames:
        print('Generate Autoencoders : {}'.format(audio_name))
        pylab.figure(figsize=FIG_SIZE_AUTOENCODERS)
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        pylab.imshow(restored_test_imgs[test_generator.filenames.index(audio_name)])
        pylab.savefig(AUTOENCODERS_STFT_PERCUSSIVE + audio_name, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        pylab.close()


def generate_valid_autoencoders():
    for audio_name in valid_generator.filenames:
        print('Generate Autoencoders : {}'.format(audio_name))
        pylab.figure(figsize=FIG_SIZE_AUTOENCODERS)
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        pylab.imshow(restored_valid_imgs[valid_generator.filenames.index(audio_name)])
        pylab.savefig(AUTOENCODERS_STFT_PERCUSSIVE + audio_name, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        pylab.close()


if __name__ == '__main__':
    k.clear_session()
    generate_train_autoencoders()
    generate_test_autoencoders()
    generate_valid_autoencoders()
