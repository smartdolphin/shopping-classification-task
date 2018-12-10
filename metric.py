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
