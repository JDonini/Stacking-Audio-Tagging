import os
import multiprocessing
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Keras Parameters
BATCH_SIZE = 4 * len(K.tensorflow_backend._get_available_gpus())
IMG_SIZE = (224, 672, 3)
TARGET_SIZE = (224, 672)
LR = 1e-3
LR_DECAY = 1e-6
MOMENTUM = 0.9
NUM_WORKERS = multiprocessing.cpu_count()
NUM_EPOCHS = 2

EXT_AUDIO = '.mp3'
EXT_IMG = '.png'
SEED = 1337
