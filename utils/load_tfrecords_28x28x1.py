import tensorflow as tf

def load_tfrecords(filename):
    #gen queue from filename list
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #return the name of file
    features = tf.parse_single_example(serialized_example,
        features={
           'label' : tf.FixedLenFeature([], tf.int64),
           'img'   : tf.FixedLenFeature([], tf.string),
        })
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label

#if __name__ == '__name__':
