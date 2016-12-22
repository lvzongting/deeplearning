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

rate = 1.0/2
#train = True ; test = False
train = False; test = True

if train: record = tf.python_io.TFRecordWriter('train28x28.tfrecord')
if test : record = tf.python_io.TFRecordWriter('test28x28.tfrecord')
for index,[root, dirs, files] in enumerate(os.walk('.')):
    if index == 0:
        classes = dirs
        continue
    flag_files = int(len(files)*rate)
    for num,sample in enumerate(files):
        #-1,28,28,1 is the size of lenet
        if train & (num > flag_files): continue
        if test  & (num <=flag_files): continue
        if train: print 'train: '+ str(index) + ':'+ str(num+1) + '/' + str(len(files))+ ':' + root + '/' +sample
        if test:  print 'test:  '+ str(index) + ':'+ str(num+1) + '/' + str(len(files))+ ':' + root + '/' +sample
        img = Image.open(root + '/' +sample)
        img = img.convert('L')
        img = img.resize((28,28)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index - 1])),
            "img"  : tf.train.Feature(bytes_list=tf.train.BytesList(value=[  img    ]))
        }))
        record.write(example.SerializeToString())
    if index == 10:
       break
record.close()
