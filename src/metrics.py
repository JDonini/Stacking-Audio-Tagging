from keras import backend as k
import tensorflow as tf


def auc_roc(y_true, y_pred):
    auc_roc = tf.metrics.auc(y_true, y_pred, curve='ROC')[1]
    k.get_session().run(tf.local_variables_initializer())
    return auc_roc


def auc_pr(y_true, y_pred):
    auc_pr = tf.metrics.auc(y_true, y_pred, curve='PR', summation_method='careful_interpolation')[1]
    k.get_session().run(tf.local_variables_initializer())
    return auc_pr


def hamming_loss(y_true, y_pred):
    y_pred = k.clip(y_pred, k.epsilon(), 1 - k.epsilon())
    return k.mean(k.sum(- y_true * k.log(y_pred) - (1 - y_true) * k.log(1 - y_pred), axis=1))


def ranking_loss(y_true, y_pred, gamma=2.0, mp=2.5, mn=0.5):
    def _loss_elem(i):
        scores, pos_label = y_pred[i], tf.cast(y_true[i][0], dtype='int32')
        pos_score = scores[pos_label]
        top_values, top_indices = tf.nn.top_k(scores, k=5)
        neg_score = tf.cond(tf.equal(top_indices[0], pos_label), lambda: top_values[1], lambda: top_values[0])
        return tf.log(1.0 + tf.exp(gamma * (mp - pos_score))) + tf.log(1.0 + tf.exp(gamma * (mn + neg_score)))
    return tf.map_fn(_loss_elem, tf.range(tf.shape(y_true)[0]), dtype=k.floatx())
