from keras import backend as K
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, hamming_loss
import tensorflow as tf
import statistics


def auc_roc(y_true, y_pred):
    auc_roc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc_roc
