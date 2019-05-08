from tqdm import tqdm
import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
sys.path.append('src/')
from generate_structure import OUT, AUDIO
sys.path.append('database/')
from config_project import EXT_IMG


def create_histogram():
    all_duration = []
    for file in tqdm(os.listdir(AUDIO)):
        y, sr = librosa.load(AUDIO + file)
        all_duration.append(librosa.get_duration(y, sr))
    plt.hist(all_duration, alpha=0.7, rwidth=0.85, bins='auto')
    plt.title('Number of Audio Samples per Time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Number of Audio Samples')
    plt.ylim(bottom=0)
    plt.savefig(OUT + 'samples-per-time' + EXT_IMG, format='png')
    plt.close()

if __name__ == '__main__':
    create_histogram()
