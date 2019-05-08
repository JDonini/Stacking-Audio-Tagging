import os
import sys
import subprocess
import numpy as np
import pandas as pd
import librosa
import librosa.display
sys.path.append('src/')
from generate_structure import BINARY_ANNOTATIONS, AUDIO
sys.path.append('database/')
from config_project import EXT_IMG, EXT_AUDIO, SEED
np.random.seed(SEED)

annotation_list, song_list, = [], []


def remove_missing_data():
    fp = open(BINARY_ANNOTATIONS, 'r+')

    content = fp.read().split('\n')
    cab = content[0].split(',')
    content = content[1:]

    for i in range(len(content)):
        content[i] = content[i].split(EXT_IMG)
        annotation_list.append(content[i][0])

    for file in os.listdir(AUDIO):
        file = file.split(EXT_AUDIO)[0]
        song_list.append(file)

    annotation_remove = [item for item in annotation_list if item not in song_list]
    song_remove = [item for item in song_list if item not in annotation_list]

    print('Remove Annotation - ', annotation_remove)
    print(len(annotation_remove))

    print('Remove Song - ', song_remove)
    print(len(song_remove))

    df = pd.read_csv(BINARY_ANNOTATIONS)
    for file in annotation_remove:
        print('Remove Annotation : {}'.format(file + EXT_IMG))
        df = df[df.song_name != file + EXT_IMG]
    df.to_csv(BINARY_ANNOTATIONS, sep=',', index=False)

if __name__ == '__main__':
    remove_missing_data()