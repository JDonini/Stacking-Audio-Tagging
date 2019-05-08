from keras import backend as K
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, hamming_loss
import tensorflow as tf
import statistics


def auc_roc(y_true, y_pred):
    auc_roc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc_roc


def top_k_categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


def subset_accuracy(y_true, y_pred):
    subsetaccuracy = 0.0
    print(y_true)
    for i in range(y_true.shape[0]):
        same = True
        for j in range(y_true.shape[1]):
            if y_true[i, j] != y_pred[i, j]:
                same = False
                break
        if same:
            subsetaccuracy += 1.0

    return subsetaccuracy/y_true.shape[0]


def hamming_loss(y_true, y_pred):
    hammingloss = 0.0
    for i in range(y_true.shape[0]):
        aux = 0.0
        for j in range(y_true.shape[1]):
            if int(y_true[i, j]) != int(y_pred[i, j]):
                aux += 1.0
        aux = aux/y_true.shape[1]
        hammingloss += aux

    return hammingloss/y_true.shape[0]


def micro_averaging(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def macro_averaging(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
