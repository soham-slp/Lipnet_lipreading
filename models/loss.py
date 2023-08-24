import tensorflow as tf
from keras import backend as K

class CTCLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CTCLoss, self).__init__()
    
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype = tf.int64)
        input_length = tf.cast(tf.shape(y_pred)[1], dtype = tf.int64)
        label_length = tf.cast(tf.shape(y_true)[1], dtype = tf.int64)
        
        input_length = input_length * tf.ones(shape = (batch_len, 1), dtype = tf.int64)
        label_length = label_length * tf.ones(shape = (batch_len, 1), dtype = tf.int64)
        
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        
        return loss