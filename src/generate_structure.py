import os
import sys
sys.path.append('config')
from config_project import FIG_SIZE, FIG_SIZE_AUTOENCODERS

fig_size = '/' + str(FIG_SIZE).replace('.', '').replace(',', '-').replace('(', '').replace(')', '').replace(' ', '')
fig_size_autoencoders = '/' + str(FIG_SIZE_AUTOENCODERS).replace('.', '').replace(',', '-').replace('(', '').replace(')', '').replace(' ', '')

pwd = os.getcwdb().decode('utf8')
database_name = os.environ["database_name"]
base = pwd + '/database/' + database_name

ANNOTATIONS = base + '/data/annotations/'
BINARY_ANNOTATIONS = ANNOTATIONS + 'binary_annotation.csv'
TRAIN_ANNOTATIONS = ANNOTATIONS + 'train.csv'
TEST_ANNOTATIONS = ANNOTATIONS + 'test.csv'
VALIDATION_ANNOTATIONS = ANNOTATIONS + 'validation.csv'

AUDIO = '/mnt/Files/Database/' + database_name + '/audios/'
AUDIO_PROCESSED = '/mnt/Files/Database/' + database_name + fig_size
AUDIO_CHROMAGRAM = AUDIO_PROCESSED + '/chromagram/'
AUDIO_MEL_SPECTROGRAM = AUDIO_PROCESSED + '/mel_spectrogram/'
AUDIO_MFCC = AUDIO_PROCESSED + '/mfcc/'
AUDIO_STFT = AUDIO_PROCESSED + '/stft/'
AUDIO_STFT_HARMONIC = AUDIO_PROCESSED + '/stft_harmonic/'
AUDIO_STFT_PERCUSSIVE = AUDIO_PROCESSED + '/stft_percussive/'

AUTOENCODERS_PROCESSED = '/mnt/Files/Database/' + database_name + fig_size_autoencoders
AUTOENCODERS_CHROMAGRAM = AUDIO_PROCESSED + '/autoencoders/chromagram/'
AUTOENCODERS_MEL_SPECTROGRAM = AUDIO_PROCESSED + '/autoencoders/mel_spectrogram/'
AUTOENCODERS_MFCC = AUDIO_PROCESSED + '/autoencoders/mfcc/'
AUTOENCODERS_STFT = AUDIO_PROCESSED + '/autoencoders/stft/'

SRC = base + '/src/'
MODEL_AUTOENCODERS = base + '/model/'
MODEL_TENSOR = base + '/model' + fig_size + '/tensorboard/'
MODEL_WEIGHTS = base + '/model' + fig_size + '/weights/'
OUT_FIRST_STAGE = base + '/out' + fig_size + '/first_stage/'
OUT_SECOND_STAGE = base + '/out' + fig_size + '/second_stage/'

MODEL_1_SRC = SRC + 'model-1/'
MODEL_1_TENSOR = MODEL_TENSOR + "model-1/"
MODEL_1_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-1/"
MODEL_1_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-1/"
MODEL_1_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-1/"

MODEL_2_SRC = SRC + 'model-2/'
MODEL_2_TENSOR = MODEL_TENSOR + "model-2/"
MODEL_2_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-2/"
MODEL_2_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-2/"
MODEL_2_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-2/"

MODEL_3_SRC = SRC + 'model-3/'
MODEL_3_TENSOR = MODEL_TENSOR + "model-3/"
MODEL_3_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-3/"
MODEL_3_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-3/"
MODEL_3_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-3/"

MODEL_4_SRC = SRC + 'model-4/'
MODEL_4_TENSOR = MODEL_TENSOR + "model-4/"
MODEL_4_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-4/"
MODEL_4_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-4/"
MODEL_4_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-4/"

MODEL_5_SRC = SRC + 'model-5/'
MODEL_5_TENSOR = MODEL_TENSOR + "model-5/"
MODEL_5_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-5/"
MODEL_5_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-5/"
MODEL_5_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-5/"

MODEL_6_SRC = SRC + 'model-6/'
MODEL_6_TENSOR = MODEL_TENSOR + "model-6/"
MODEL_6_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-6/"
MODEL_6_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-6/"
MODEL_6_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-6/"

MODEL_7_SRC = SRC + 'model-7/'
MODEL_7_TENSOR = MODEL_TENSOR + "model-7/"
MODEL_7_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-7/"
MODEL_7_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-7/"
MODEL_7_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-7/"

MODEL_8_SRC = SRC + 'model-8/'
MODEL_8_TENSOR = MODEL_TENSOR + "model-8/"
MODEL_8_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-8/"
MODEL_8_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-8/"
MODEL_8_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-8/"

MODEL_9_SRC = SRC + 'model-9/'
MODEL_9_TENSOR = MODEL_TENSOR + "model-9/"
MODEL_9_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-9/"
MODEL_9_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-9/"
MODEL_9_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-9/"

MODEL_10_SRC = SRC + 'model-10/'
MODEL_10_TENSOR = MODEL_TENSOR + "model-10/"
MODEL_10_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-10/"
MODEL_10_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-10/"
MODEL_10_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-10/"

MODEL_11_SRC = SRC + 'model-11/'
MODEL_11_TENSOR = MODEL_TENSOR + "model-11/"
MODEL_11_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-11/"
MODEL_11_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-11/"
MODEL_11_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-11/"

MODEL_12_SRC = SRC + 'model-12/'
MODEL_12_TENSOR = MODEL_TENSOR + "model-12/"
MODEL_12_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-12/"
MODEL_12_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-12/"
MODEL_12_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-12/"

MODEL_13_SRC = SRC + 'model-13/'
MODEL_13_TENSOR = MODEL_TENSOR + "model-13/"
MODEL_13_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-13/"
MODEL_13_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-13/"
MODEL_13_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-13/"

MODEL_14_SRC = SRC + 'model-14/'
MODEL_14_TENSOR = MODEL_TENSOR + "model-14/"
MODEL_14_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-14/"
MODEL_14_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-14/"
MODEL_14_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-14/"

MODEL_15_SRC = SRC + 'model-15/'
MODEL_15_TENSOR = MODEL_TENSOR + "model-15/"
MODEL_15_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-15/"
MODEL_15_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-15/"
MODEL_15_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-15/"

MODEL_16_SRC = SRC + 'model-16/'
MODEL_16_TENSOR = MODEL_TENSOR + "model-16/"
MODEL_16_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-16/"
MODEL_16_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-16/"
MODEL_16_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-16/"

MODEL_17_SRC = SRC + 'model-17/'
MODEL_17_TENSOR = MODEL_TENSOR + "model-17/"
MODEL_17_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-17/"
MODEL_17_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-17/"
MODEL_17_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-17/"

MODEL_18_SRC = SRC + 'model-18/'
MODEL_18_TENSOR = MODEL_TENSOR + "model-18/"
MODEL_18_WEIGHTS_FINAL = MODEL_WEIGHTS + "model-18/"
MODEL_18_OUT_FIRST_STAGE = OUT_FIRST_STAGE + "model-18/"
MODEL_18_OUT_SECOND_STAGE = OUT_SECOND_STAGE + "model-18/"

list_dir = [ANNOTATIONS, AUDIO, AUDIO_MEL_SPECTROGRAM, AUDIO_MFCC, AUDIO_MEL_SPECTROGRAM, 
            AUDIO_STFT, AUDIO_STFT_HARMONIC, AUDIO_STFT_PERCUSSIVE, AUDIO_CHROMAGRAM,
            AUTOENCODERS_CHROMAGRAM, AUTOENCODERS_MEL_SPECTROGRAM, AUTOENCODERS_MFCC,
            AUTOENCODERS_STFT, MODEL_AUTOENCODERS,
            MODEL_1_SRC, MODEL_1_TENSOR, MODEL_1_WEIGHTS_FINAL,
            MODEL_1_OUT_FIRST_STAGE, MODEL_1_OUT_SECOND_STAGE,
            MODEL_2_SRC, MODEL_2_TENSOR, MODEL_2_WEIGHTS_FINAL,
            MODEL_2_OUT_FIRST_STAGE, MODEL_2_OUT_SECOND_STAGE,
            MODEL_3_SRC, MODEL_3_TENSOR, MODEL_3_WEIGHTS_FINAL,
            MODEL_3_OUT_FIRST_STAGE, MODEL_3_OUT_SECOND_STAGE,
            MODEL_4_SRC, MODEL_4_TENSOR, MODEL_4_WEIGHTS_FINAL,
            MODEL_4_OUT_FIRST_STAGE, MODEL_4_OUT_SECOND_STAGE,
            MODEL_5_SRC, MODEL_5_TENSOR, MODEL_5_WEIGHTS_FINAL,
            MODEL_5_OUT_FIRST_STAGE, MODEL_5_OUT_SECOND_STAGE,
            MODEL_6_SRC, MODEL_6_TENSOR, MODEL_6_WEIGHTS_FINAL,
            MODEL_6_OUT_FIRST_STAGE, MODEL_6_OUT_SECOND_STAGE,
            MODEL_7_SRC, MODEL_7_TENSOR, MODEL_7_WEIGHTS_FINAL,
            MODEL_7_OUT_FIRST_STAGE, MODEL_7_OUT_SECOND_STAGE,
            MODEL_8_SRC, MODEL_8_TENSOR, MODEL_8_WEIGHTS_FINAL,
            MODEL_8_OUT_FIRST_STAGE, MODEL_8_OUT_SECOND_STAGE,
            MODEL_9_SRC, MODEL_9_TENSOR, MODEL_9_WEIGHTS_FINAL,
            MODEL_9_OUT_FIRST_STAGE, MODEL_9_OUT_SECOND_STAGE,
            MODEL_10_SRC, MODEL_10_TENSOR, MODEL_10_WEIGHTS_FINAL,
            MODEL_10_OUT_FIRST_STAGE, MODEL_10_OUT_SECOND_STAGE,
            MODEL_11_SRC, MODEL_11_TENSOR, MODEL_11_WEIGHTS_FINAL,
            MODEL_11_OUT_FIRST_STAGE, MODEL_11_OUT_SECOND_STAGE,
            MODEL_12_SRC, MODEL_12_TENSOR, MODEL_12_WEIGHTS_FINAL,
            MODEL_12_OUT_FIRST_STAGE, MODEL_12_OUT_SECOND_STAGE,
            MODEL_13_SRC, MODEL_13_TENSOR, MODEL_13_WEIGHTS_FINAL,
            MODEL_13_OUT_FIRST_STAGE, MODEL_13_OUT_SECOND_STAGE,
            MODEL_14_SRC, MODEL_14_TENSOR, MODEL_14_WEIGHTS_FINAL,
            MODEL_14_OUT_FIRST_STAGE, MODEL_14_OUT_SECOND_STAGE,
            MODEL_15_SRC, MODEL_15_TENSOR, MODEL_15_WEIGHTS_FINAL,
            MODEL_15_OUT_FIRST_STAGE, MODEL_15_OUT_SECOND_STAGE,
            MODEL_16_SRC, MODEL_16_TENSOR, MODEL_16_WEIGHTS_FINAL,
            MODEL_16_OUT_FIRST_STAGE, MODEL_16_OUT_SECOND_STAGE,
            MODEL_17_SRC, MODEL_17_TENSOR, MODEL_17_WEIGHTS_FINAL,
            MODEL_17_OUT_FIRST_STAGE, MODEL_17_OUT_SECOND_STAGE,
            MODEL_18_SRC, MODEL_18_TENSOR, MODEL_18_WEIGHTS_FINAL,
            MODEL_18_OUT_FIRST_STAGE, MODEL_18_OUT_SECOND_STAGE
            ]

for fold in list_dir:
    os.makedirs(fold, exist_ok=True)
