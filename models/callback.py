import tensorflow as tf
from keras import backend as K

class ProduceExample(tf.keras.callbacks.Callback): 
    def __init__(self, dataset) -> None:
        super(ProduceExample, self).__init__()
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs = None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = K.ctc_decode(yhat, [75, 75], greedy = False)[0][0].numpy()
        for x in range(len(yhat)):           
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~' * 100)