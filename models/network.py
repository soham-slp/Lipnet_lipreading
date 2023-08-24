import tensorflow as tf
from keras import backend as K

class LipNet(tf.keras.Model):
    def __init__(self, output_size):
        super(LipNet, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv3D(64, (3, 5, 5), 
                                            strides = (1, 1, 1), 
                                            use_bias = False, 
                                            padding = 'same', 
                                           name = 'conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name = 'bn1')
        self.act1 = tf.keras.layers.Activation('relu')
        self.drop1 = tf.keras.layers.SpatialDropout3D(0.3)
        self.pool1 = tf.keras.layers.MaxPooling3D(pool_size = (1, 2, 2), strides = (1, 2, 2), name = 'pool1')
        
        self.conv2 = tf.keras.layers.Conv3D(64, (3, 5, 5), 
                                            strides = (1, 1, 1), 
                                            use_bias = False, 
                                            padding = 'same', 
                                           name = 'conv2')
        self.bn2 = tf.keras.layers.BatchNormalization(name = 'bn2')
        self.act2 = tf.keras.layers.Activation('relu')
        self.drop2 = tf.keras.layers.SpatialDropout3D(0.3)
        self.pool2 = tf.keras.layers.MaxPooling3D(pool_size = (1, 2, 2), strides = (1, 2, 2), name = 'pool2')
        
        self.conv3 = tf.keras.layers.Conv3D(64, (3, 5, 5), 
                                            strides = (1, 1, 1), 
                                            use_bias = False, 
                                            padding = 'same', 
                                           name = 'conv3')
        self.bn3 = tf.keras.layers.BatchNormalization(name = 'bn3')
        self.act3 = tf.keras.layers.Activation('relu')
        self.drop3 = tf.keras.layers.SpatialDropout3D(0.3)
        self.pool3 = tf.keras.layers.MaxPooling3D(pool_size = (1, 2, 2), strides = (1, 2, 2), name = 'pool3')
        
        self.res1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        
        self.gru1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128,
                                return_sequences=True,
                                kernel_initializer='Orthogonal',
                                name='gru1'),
            merge_mode='concat'
        )
        
        self.gru2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, 
                                return_sequences=True, 
                                kernel_initializer='Orthogonal', 
                                name='gru2'), 
            merge_mode='concat'
        )
        
        self.output_layer = tf.keras.layers.Dense(output_size, activation = 'softmax')
    
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.pool3(x)
        
        x = self.res1(x)
        
        x = self.gru1(x)
        x = self.gru2(x)
        
        x = self.output_layer(x)
        
        return x