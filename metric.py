import numpy as np
import tensorflow as tf
import keras.backend as K


def fbeta_score_macro(y_true, y_pred, beta=1, threshold=0.5):
    y_true = K.cast(y_true, 'float')
    y_pred = K.cast(K.greater(K.cast(y_pred, 'float'), threshold), 'float')

    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def arena_score(y_true, y_pred, vocab_matrix=None):
    assert vocab_matrix is not None
    vocab = K.variable(value=vocab_matrix)
    y_pred = K.one_hot(K.argmax(y_pred, axis=-1), vocab_matrix.shape[0])

    true_mat = K.dot(y_true, vocab)
    pred_mat = K.dot(y_pred, vocab)

    filtering = K.variable(value=np.array([-1] * 4).reshape(1, 4),
                           dtype='int32')
    score = K.variable(value=np.array([[1, 1.2, 1.3, 1.4]]).reshape(1, 4),
                       dtype='float32')

    filtering = K.cast(filtering, 'float32')
    true_mask = K.cast(K.greater(true_mat, filtering), 'int32')
    pred_mask = K.cast(K.greater(pred_mat, filtering), 'int32')

    mask = K.clip(true_mask + pred_mask, 0, 1)

    same = tf.reduce_all(tf.equal(K.reshape(true_mat, (-1, 1)),
                                  K.reshape(pred_mat, (-1, 1))), axis=-1)

    target = K.reshape(K.cast(same, 'int32'), (-1, 4)) * mask
    target = K.cast(target, 'float32')
    return K.mean(K.sum(target * score, axis=-1) * 1./4)
