from keras import backend as K
import tensorflow as tf
import numpy as np


def auc_roc(y_true, y_pred):
    auc_roc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc_roc


def hamming_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


def ranking_loss(y_true, y_pred, gamma=2.0, mp=2.5, mn=0.5):
    def _loss_elem(i):
        scores, pos_label = y_pred[i], tf.cast(y_true[i][0], dtype='int32')
        pos_score = scores[pos_label]
        top_values, top_indices = tf.nn.top_k(scores, k=5)
        neg_score = tf.cond(tf.equal(top_indices[0], pos_label), lambda: top_values[1], lambda: top_values[0])
        return tf.log(1.0 + tf.exp(gamma * (mp - pos_score))) + tf.log(1.0 + tf.exp(gamma * (mn + neg_score)))
    return tf.map_fn(_loss_elem, tf.range(tf.shape(y_true)[0]), dtype=K.floatx())


class SubsetAccuracy(Callback):
    def on_train_begin(self, logs={}):
        self.accuracy_score = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(
            self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.confusion.append(confusion_matrix(targ, predict))
        self.precision.append(precision_score(targ, predict))
        self.recall.append(recall_score(targ, predict))
        self.f1s.append(f1_score(targ, predict))
        self.kappa.append(cohen_kappa_score(targ, predict))
