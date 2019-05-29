from tqdm import tqdm
import os
import sys
import subprocess
import numpy as np
import pandas as pd
import librosa
import librosa.display
sys.path.append('src/')
from generate_structure import TAG_ANNOTATIONS, BINARY_ANNOTATIONS, AUDIO
sys.path.append('database/')
from config_project import EXT_AUDIO, AUDIO_THRESHOLD, EXT_IMG, SEED

np.random.seed(SEED)

column_name = []
dict_names, dict_tags, folders_name = {}, {}, {}


def remove_short_audio():
    print('Removing Short Audios')
    for file in tqdm(os.listdir(AUDIO)):
        y, sr = librosa.load(AUDIO + file)
        audio_duration = librosa.core.get_duration(y=y, sr=sr)
        if audio_duration < AUDIO_THRESHOLD:
            with open(TAG_ANNOTATIONS, 'rt') as lines:
                for line in lines:
                    audio_name, _ = file.split(EXT_AUDIO)
                    if line.startswith(audio_name):
                        subprocess.call(['sed', '-i', '/.*' + str(audio_name) + '.*/d', TAG_ANNOTATIONS])
                    if os.path.isfile(AUDIO + file):
                        os.remove(AUDIO + file)


def create_dict_annotations():
    with open(TAG_ANNOTATIONS) as f:
        for row in f:
            song_name, tag_name = row.strip().split('\t')
            song_name = song_name + EXT_IMG
            if tag_name not in column_name:
                column_name.append(tag_name)
            if tag_name not in dict_tags.keys():
                dict_tags.setdefault(tag_name, len(dict_tags.keys()) + 1)
            if song_name not in dict_names.keys():
                dict_names.setdefault(song_name, [tag_name])
            else:
                dict_names[song_name].append(tag_name)
    dict_mapping = {k: [] for k in dict_names}
    for tag_key, tag_value in dict_tags.items():
        for song_name, value in dict_names.items():
            for v in value:
                if v == tag_key:
                    dict_mapping[song_name].append(tag_value)
    return dict_mapping


def create_train_test_validation():
    for song_name, tag_value in create_dict_annotations().items():
        tag_value = '-'.join(str(x) for x in tag_value)
        folders_name.setdefault(song_name, tag_value)
    return folders_name


def create_annotations():
    df_binary = pd.DataFrame(columns=column_name, index=dict_names.keys())
    df_binary.index.name = 'song_name'
    df_binary[:] = 0
    for key, items in dict_names.items():
        print('One Hot Annotation: {}'.format(key))
        df_binary.loc[key, items] = 1
    df_binary.to_csv(BINARY_ANNOTATIONS, sep=',')


if __name__ == '__main__':
    remove_short_audio()
    create_dict_annotations()
    create_annotations()
