import os
import tensorflow as tf
from PIL import Image

'''
tree of dataset dirs
0 -- img1.jpg
     img2.jpg
     img3.jpg
     ...
1 -- img1.jpg
     img2.jpg
     ...
2 -- ...
...
'''

record = tf.python_io.TFRecordWriter('train.tfrecord')
for index,[root, dirs, files] in enumerate(os.walk('.')):
    if index == 0:
        classes = dirs
        continue
    for sample in files:
        #227,227 is the size of Alexnet
        print root + '/' +sample
        img = Image.open(root + '/' +sample)
        img = img.resize((227,227)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            "img"  : tf.train.Feature(bytes_list=tf.train.BytesList(value=[  img]))
        }))
        record.write(example.SerializeToString())
    #if index == 2:
    #    break
record.close()
