import sys
import os
import pandas as pd
import tensorflow as tf
import pylab
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras import backend as k
sys.path.append('src')
from generate_structure import MODEL_AUTOENCODERS, BINARY_ANNOTATIONS, VALIDATION_ANNOTATIONS, AUDIO_MFCC, \
 AUTOENCODERS_MFCC
sys.path.append('database')
from config_project import SEED, BATCH_SIZE, TARGET_SIZE, FIG_SIZE_AUTOENCODERS

np.random.seed(SEED)
tf.set_random_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

columns = pd.read_csv(VALIDATION_ANNOTATIONS).columns[1:].tolist()
datagen = ImageDataGenerator(rescale=1./255)

predict_generator = datagen.flow_from_dataframe(
    dataframe=pd.read_csv(BINARY_ANNOTATIONS, skiprows=range(1, 24000), nrows=4000),
    directory=AUDIO_MFCC,
    x_col='song_name',
    y_col=columns,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode='input',
    target_size=TARGET_SIZE
)

model = load_model(MODEL_AUTOENCODERS + 'model_mfcc.h5')


predict_generator.reset()
restored_predict_imgs = model.predict_generator(predict_generator,
                                                steps=predict_generator.n / predict_generator.batch_size, verbose=1)


def generate_predict_autoencoders():
    for audio_name in predict_generator.filenames:
        print('Generate Autoencoders : {}'.format(audio_name))
        pylab.figure(figsize=FIG_SIZE_AUTOENCODERS)
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        pylab.imshow(restored_predict_imgs[predict_generator.filenames.index(audio_name)])
        pylab.savefig(AUTOENCODERS_MFCC + audio_name, bbox_inches=None, pad_inches=0, format='png', dpi=100)
        pylab.close()


if __name__ == '__main__':
    k.clear_session()
    generate_predict_autoencoders()
