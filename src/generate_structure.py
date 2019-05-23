import os
import sys
sys.path.append('database')
from config_project import FIG_SIZE

fig_size = '/' + str(FIG_SIZE).replace('.', '').replace(',', '-').replace('(', '').replace(')', '').replace(' ', '')

pwd = os.getcwdb().decode('utf8')
database_name = os.environ["database_name"]
BASE = pwd + '/database/' + database_name

ANNOTATIONS = BASE + '/data/annotations/'
BINARY_ANNOTATIONS = ANNOTATIONS + 'binary_annotation.csv'
TRAIN_ANNOTATIONS = ANNOTATIONS + 'train.csv'
TEST_ANNOTATIONS = ANNOTATIONS + 'test.csv'
VALIDATION_ANNOTATIONS = ANNOTATIONS + 'validation.csv'

AUDIO = '/mnt/Files/Database/' + database_name + '/audio/'
AUDIO_PROCESSED = '/mnt/Files/Database/' + database_name + fig_size
AUDIO_MEL_SPECTROGRAM = AUDIO_PROCESSED + '/mel_spectrogram/'
AUDIO_MFCC = AUDIO_PROCESSED + '/mfcc/'
AUDIO_STFT_HARMONIC = AUDIO_PROCESSED + '/stft_harmonic/'
AUDIO_STFT_PERCUSSIVE = AUDIO_PROCESSED + '/stft_percussive/'
AUDIO_WAVEFORM = AUDIO_PROCESSED + '/waveform/'
AUDIO_CHROMAGRAM = AUDIO_PROCESSED + '/chromagram/'

SRC = BASE + '/src/'
MODEL_TENSOR = BASE + '/model/tensorboard/'
MODEL_WEIGHTS_FINAL = BASE + '/model/weights_final/'
MODEL_WEIGTHS_PER_EPOCHS = BASE + '/model/weights_per_epochs/'
OUT_FIRST_STAGE = BASE + '/out/first_stage/'
OUT_SECOND_STAGE = BASE + '/out/second_stage/'

# Model CNN-CNN-SIMPLE
MODEL_1_SRC = SRC + 'model-1/'
MODEL_1_TENSOR = MODEL_TENSOR + "model-1/"
MODEL_1_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-1/"
MODEL_1_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-1/"
MODEL_1_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-1/"
MODEL_1_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-1/"

# Model CNN-CNN-ROBOUST
MODEL_2_SRC = SRC + 'model-2/'
MODEL_2_TENSOR = MODEL_TENSOR + "model-2/"
MODEL_2_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-2/"
MODEL_2_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-2/"
MODEL_2_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-2/"
MODEL_2_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-2/"

# Model CNN-CNN-MERGE-SIMPLE
MODEL_3_SRC = SRC + 'model-3/'
MODEL_3_TENSOR = MODEL_TENSOR + "model-3/"
MODEL_3_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-3/"
MODEL_3_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-3/"
MODEL_3_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-3/"
MODEL_3_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-3/"

# Model CNN-CNN-MERGE-ROBOUST
MODEL_4_SRC = SRC + 'model-4/'
MODEL_4_TENSOR = MODEL_TENSOR + "model-4/"
MODEL_4_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-4/"
MODEL_4_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-4/"
MODEL_4_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-4/"
MODEL_4_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-4/"

# Model CNN-CNN-Learning-Weights
MODEL_5_SRC = SRC + 'model-5/'
MODEL_5_TENSOR = MODEL_TENSOR + "model-5/"
MODEL_5_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-5/"
MODEL_5_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-5/"
MODEL_5_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-5/"
MODEL_5_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-5/"

MODEL_6_SRC = SRC + 'model-6/'
MODEL_6_TENSOR = MODEL_TENSOR + "model-6/"
MODEL_6_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-6/"
MODEL_6_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-6/"
MODEL_6_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-6/"
MODEL_6_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-6/"

MODEL_7_SRC = SRC + 'model-7/'
MODEL_7_TENSOR = MODEL_TENSOR + "model-7/"
MODEL_7_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-7/"
MODEL_7_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-7/"
MODEL_7_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-7/"
MODEL_7_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-7/"

MODEL_8_SRC = SRC + 'model-8/'
MODEL_8_TENSOR = MODEL_TENSOR + "model-8/"
MODEL_8_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-8/"
MODEL_8_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-8/"
MODEL_8_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-8/"
MODEL_8_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-8/"

MODEL_9_SRC = SRC + 'model-9/'
MODEL_9_TENSOR = MODEL_TENSOR + "model-9/"
MODEL_9_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-9/"
MODEL_9_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-9/"
MODEL_9_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-9/"
MODEL_9_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-9/"

MODEL_10_SRC = SRC + 'model-10/'
MODEL_10_TENSOR = MODEL_TENSOR + "model-10/"
MODEL_10_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-10/"
MODEL_10_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-10/"
MODEL_10_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-10/"
MODEL_10_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-10/"

MODEL_11_SRC = SRC + 'model-11/'
MODEL_11_TENSOR = MODEL_TENSOR + "model-11/"
MODEL_11_WEIGHTS_FINAL = MODEL_WEIGHTS_FINAL + "model-11/"
MODEL_11_WEIGTHS_PER_EPOCHS = MODEL_WEIGTHS_PER_EPOCHS + "model-11/"
MODEL_11_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-11/"
MODEL_11_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-11/"

list_dir = [ANNOTATIONS, AUDIO, AUDIO_MEL_SPECTROGRAM, AUDIO_MFCC, AUDIO_MEL_SPECTROGRAM,
            AUDIO_STFT_HARMONIC, AUDIO_STFT_PERCUSSIVE, AUDIO_WAVEFORM, AUDIO_CHROMAGRAM,
            MODEL_1_SRC, MODEL_1_TENSOR, MODEL_1_WEIGHTS_FINAL, MODEL_1_WEIGTHS_PER_EPOCHS,
            MODEL_1_OUT_FIRST_STAGE, MODEL_1_OUT_SECOND_STAGE,
            MODEL_2_SRC, MODEL_2_TENSOR, MODEL_2_WEIGHTS_FINAL, MODEL_2_WEIGTHS_PER_EPOCHS,
            MODEL_2_OUT_FIRST_STAGE, MODEL_2_OUT_SECOND_STAGE,
            MODEL_3_SRC, MODEL_3_TENSOR, MODEL_3_WEIGHTS_FINAL, MODEL_3_WEIGTHS_PER_EPOCHS,
            MODEL_3_OUT_FIRST_STAGE, MODEL_3_OUT_SECOND_STAGE,
            MODEL_4_SRC, MODEL_4_TENSOR, MODEL_4_WEIGHTS_FINAL, MODEL_4_WEIGTHS_PER_EPOCHS,
            MODEL_4_OUT_FIRST_STAGE, MODEL_4_OUT_SECOND_STAGE,
            MODEL_5_SRC, MODEL_5_TENSOR, MODEL_5_WEIGHTS_FINAL, MODEL_5_WEIGTHS_PER_EPOCHS,
            MODEL_5_OUT_FIRST_STAGE, MODEL_5_OUT_SECOND_STAGE,
            MODEL_6_SRC, MODEL_6_TENSOR, MODEL_6_WEIGHTS_FINAL, MODEL_6_WEIGTHS_PER_EPOCHS,
            MODEL_6_OUT_FIRST_STAGE, MODEL_6_OUT_SECOND_STAGE,
            MODEL_7_SRC, MODEL_7_TENSOR, MODEL_7_WEIGHTS_FINAL, MODEL_7_WEIGTHS_PER_EPOCHS,
            MODEL_7_OUT_FIRST_STAGE, MODEL_7_OUT_SECOND_STAGE,
            MODEL_8_SRC, MODEL_8_TENSOR, MODEL_8_WEIGHTS_FINAL, MODEL_8_WEIGTHS_PER_EPOCHS,
            MODEL_8_OUT_FIRST_STAGE, MODEL_8_OUT_SECOND_STAGE,
            MODEL_9_SRC, MODEL_9_TENSOR, MODEL_9_WEIGHTS_FINAL, MODEL_9_WEIGTHS_PER_EPOCHS,
            MODEL_9_OUT_FIRST_STAGE, MODEL_9_OUT_SECOND_STAGE,
            MODEL_10_SRC, MODEL_10_TENSOR, MODEL_10_WEIGHTS_FINAL, MODEL_10_WEIGTHS_PER_EPOCHS,
            MODEL_10_OUT_FIRST_STAGE, MODEL_10_OUT_SECOND_STAGE,
            MODEL_11_SRC, MODEL_11_TENSOR, MODEL_11_WEIGHTS_FINAL, MODEL_11_WEIGTHS_PER_EPOCHS,
            MODEL_11_OUT_FIRST_STAGE, MODEL_11_OUT_SECOND_STAGE,
            ]

for fold in list_dir:
    os.makedirs(fold, exist_ok=True)
