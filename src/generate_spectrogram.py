from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import sys
import os
import pylab
import librosa
import librosa.display
sys.path.append('src/')
from generate_structure import AUDIO, AUDIO_MFCC, AUDIO_MEL_SPECTROGRAM, \
 AUDIO_WAVEFORM, AUDIO_STFT_PERCUSSIVE, AUDIO_STFT_HARMONIC, AUDIO_CHROMAGRAM
sys.path.append('database')
from config_project import EXT_AUDIO, EXT_IMG, FIG_SIZE, N_MELS, SR, OFFSET, DURATION


pbar = tqdm(total=len(os.listdir(AUDIO)))


def create_mel_spectrogram(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    S = librosa.feature.melspectrogram(y, sr=SR, n_mels=N_MELS)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(log_S, sr=SR)
    pylab.savefig(AUDIO_MEL_SPECTROGRAM + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_stft_harmonic(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    y_harmonic, _ = librosa.effects.hpss(y)
    S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=SR)
    log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
    librosa.display.specshow(log_Sh, sr=SR)
    pylab.savefig(AUDIO_STFT_HARMONIC + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_stft_percussive(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    _, y_percussive = librosa.effects.hpss(y)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=SR)
    log_Sp = librosa.power_to_db(S_percussive, ref=np.max)
    librosa.display.specshow(log_Sp, sr=SR)
    pylab.savefig(AUDIO_STFT_PERCUSSIVE + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_chromagram(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    y_harmonic, _ = librosa.effects.hpss(y)
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=SR)
    librosa.display.specshow(C, sr=SR, vmin=0, vmax=1)
    pylab.savefig(AUDIO_CHROMAGRAM + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_mfcc(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS)
    log_S = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(librosa.feature.mfcc(S=log_S, n_mfcc=13))
    pylab.savefig(AUDIO_MFCC + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def create_waveform(y, audio_name):
    pylab.figure(figsize=FIG_SIZE)
    pylab.axis('off')
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
    librosa.display.waveplot(y=y, sr=SR, alpha=0.8)
    pylab.savefig(AUDIO_WAVEFORM + audio_name + EXT_IMG, bbox_inches=None, pad_inches=0, format='png', dpi=100)
    pylab.close()


def update(*a):
    pbar.update()


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for full_dir, file in [[(os.path.join(root, f)), (f)] for root, dirs, files in os.walk(AUDIO) for f in files]:
        y, _ = librosa.load(full_dir, offset=OFFSET, duration=DURATION)
        pool.apply_async(create_chromagram, args=(y, file.split(EXT_AUDIO)[0]), callback=update)
        pool.apply_async(create_mel_spectrogram, args=(y, file.split(EXT_AUDIO)[0]))
        pool.apply_async(create_mfcc, args=(y, file.split(EXT_AUDIO)[0]))
        pool.apply_async(create_stft_harmonic, args=(y, file.split(EXT_AUDIO)[0]))
        pool.apply_async(create_stft_percussive, args=(y, file.split(EXT_AUDIO)[0]))
        pool.apply_async(create_waveform, args=(y, file.split(EXT_AUDIO)[0]))
    pool.close()
    pool.join()
