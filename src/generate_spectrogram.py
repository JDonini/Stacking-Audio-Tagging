from tqdm import tqdm
from multiprocessing import Process, Pool, cpu_count
from glob import glob
import os
import numpy as np
import sys
import pylab
import librosa
import librosa.display
import matplotlib.pyplot as plt
sys.path.append('src/')
from generate_structure import AUDIO, AUDIO_MFCC, AUDIO_MEL_SPECTROGRAM, \
 AUDIO_STFT_PERCUSSIVE, AUDIO_STFT_HARMONIC, AUDIO_CHROMAGRAM, AUDIO_STFT
sys.path.append('database')
from config_project import EXT_AUDIO, EXT_IMG, FIG_SIZE, N_MELS, SR, OFFSET, DURATION

plt.rcParams.update({'figure.max_open_warning': 0})

pbar = tqdm(total=len(os.listdir(AUDIO)))


def create_chromagram(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    y_harmonic, _ = librosa.effects.hpss(y)
    librosa.display.specshow(librosa.feature.chroma_cqt(y=y_harmonic, sr=SR), sr=SR, vmin=0, vmax=1)
    pylab.savefig(AUDIO_CHROMAGRAM + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_mel_spectrogram(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y, sr=SR, n_mels=N_MELS), ref=np.max), sr=SR)
    pylab.savefig(AUDIO_MEL_SPECTROGRAM + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_mfcc(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    log_s = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS), ref=np.max)
    librosa.display.specshow(librosa.feature.mfcc(S=log_s, n_mfcc=13))
    pylab.savefig(AUDIO_MFCC + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_stft(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max(np.abs(librosa.stft(y)))), y_axis='log')
    pylab.savefig(AUDIO_STFT + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_stft_harmonic(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    y_harmonic, _ = librosa.decompose.hpss(librosa.stft(y))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(y_harmonic), ref=np.max(np.abs(librosa.stft(y)))), y_axis='log')
    pylab.savefig(AUDIO_STFT_HARMONIC + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_stft_percussive(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    _, y_percussive = librosa.decompose.hpss(librosa.stft(y))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(y_percussive), ref=np.max(np.abs(librosa.stft(y)))), y_axis='log')
    pylab.savefig(AUDIO_STFT_PERCUSSIVE + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def update(*a):
    pbar.update()


if __name__ == '__main__':
    pool = Pool(processes=cpu_count())
    for audio_file in glob(AUDIO + EXT_AUDIO):
        y, _ = librosa.load(audio_file, offset=OFFSET, duration=DURATION)
        pool.apply_async(create_chromagram, args=(y, os.path.splitext(os.path.basename(audio_file))[0]), callback=update)
        pool.apply_async(create_mel_spectrogram, args=(y, os.path.splitext(os.path.basename(audio_file))[0]))
        pool.apply_async(create_mfcc, args=(y, os.path.splitext(os.path.basename(audio_file))[0]))
        pool.apply_async(create_stft, args=(y, os.path.splitext(os.path.basename(audio_file))[0]))
        pool.apply_async(create_stft_harmonic, args=(y, os.path.splitext(os.path.basename(audio_file))[0]))
        pool.apply_async(create_stft_percussive, args=(y, os.path.splitext(os.path.basename(audio_file))[0]))
    pool.close()
    pool.join()
