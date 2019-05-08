import multiprocessing

# Keras Parameters
BATCH_SIZE = 8
IMG_SIZE = (224, 672, 3)
TARGET_SIZE = (224, 672)
LR = 1e-3
LR_DECAY = 1e-6
MOMENTUM = 0.9
NUM_WORKERS = multiprocessing.cpu_count()
NUM_EPOCHS = 200

EXT_AUDIO = '.mp3'
EXT_IMG = '.png'
SEED = 1337
