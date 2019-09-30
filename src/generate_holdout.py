from sklearn.utils import shuffle
import sys
import numpy as np
import pandas as pd
sys.path.append('src')
from generate_structure import BINARY_ANNOTATIONS, TRAIN_ANNOTATIONS, TEST_ANNOTATIONS, VALIDATION_ANNOTATIONS
sys.path.append('config')
from config_project import SEED

np.random.seed(SEED)

df_tag_annotations = pd.read_csv(BINARY_ANNOTATIONS)
df_tag_annotations = shuffle(df_tag_annotations)

labels = list(df_tag_annotations.columns.values[0::])
samples_count = len(df_tag_annotations[labels[0]])

train_count = int(samples_count * 0.6)
test_count = int(samples_count * 0.3)

data_train = pd.DataFrame()
data_train = df_tag_annotations[labels[::]][:train_count]
data_train.to_csv(TRAIN_ANNOTATIONS, index=False)

data_test = pd.DataFrame()
data_test = pd.DataFrame(df_tag_annotations[labels[::]][train_count: train_count + test_count])
data_test.to_csv(TEST_ANNOTATIONS, index=False)

data_validation = pd.DataFrame()
data_validation = pd.DataFrame(df_tag_annotations[labels[::]][train_count + test_count:])
data_validation.to_csv(VALIDATION_ANNOTATIONS, index=False)
