import tensorflow as tf 
#from utils.load_tfrecord import *
#img, label = load_tfrecord("train.tfrecord")

def load_tfrecord(filename):                                                                                                                      
    #gen queue from filename list
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #return the name of file
    features = tf.parse_single_example(serialized_example,
        features={
           'label'   : tf.FixedLenFeature([], tf.int64),
           'img_raw' : tf.FixedLenFeature([], tf.string),
        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label
