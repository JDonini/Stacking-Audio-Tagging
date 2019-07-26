from tqdm import tqdm
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
from config_project import EXT_AUDIO, AUDIO_THRESHOLD, EXT_IMG, SEED

np.random.seed(SEED)

column_name = []
dict_names, dict_tags, folders_name = {}, {}, {}

df_tag_annotations = pd.read_csv(BINARY_ANNOTATIONS)
df_tag_annotations = df_tag_annotations.drop(df_tag_annotations.columns[0], axis=1)
N = len(df_tag_annotations)


def create_annotations():
    df_binary = pd.DataFrame(columns=column_name, index=dict_names.keys())
    df_binary.index.name = 'song_name'
    df_binary[:] = 0
    for key, items in dict_names.items():
        print('One Hot Annotation: {}'.format(key))
        df_binary.loc[key, items] = 1
    df_binary.to_csv(BINARY_ANNOTATIONS, sep=',')


def cardinality():
    sum_y_i = df_tag_annotations.apply(lambda row: sum(row[::] == 1), axis=1).sum()
    print('Cardinality: {:.4f}'.format(sum_y_i / N))


def density():
    labels = len(list(df_tag_annotations.columns.values[0::]))
    sum_y_i_l = df_tag_annotations.apply(lambda row: sum(row[::] == 1)/labels, axis=1).sum()
    print('Density: {:.4f}'.format(sum_y_i_l / N))


if __name__ == '__main__':
    create_annotations()
    cardinality()
    density()
