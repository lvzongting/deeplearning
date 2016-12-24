import tensorflow as tf


for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    image = example.features.feature['img'  ].bytes_list.value
    label = example.features.feature['label'].int64_list.value

    print label, len(image)
