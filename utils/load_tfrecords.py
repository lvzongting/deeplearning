import tensorflow as tf 
#from utils.load_tfrecord import *
#img, label = load_tfrecord("train.tfrecord")
#img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                batch_size=30, capacity=2000,
#                                                min_after_dequeue=1000)
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
#    threads = tf.train.start_queue_runners(sess=sess)
#    for i in range(3):
#        val, l= sess.run([img_batch, label_batch])
#        print(val.shape, l)

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
    img = tf.reshape(img, [227, 227, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label
