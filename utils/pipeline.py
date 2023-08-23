import tensorflow as tf
import numpy as np
from .loading import *


def mappable_function(path:str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

def create_pipeline():
    data = tf.data.Dataset.list_files('./data/s1/*.mpg')\
        .shuffle(500).map(mappable_function)\
        .padded_batch(2, padded_shapes = ([75, None, None, None], [40]))\
        .prefetch(tf.data.AUTOTUNE)
    return data