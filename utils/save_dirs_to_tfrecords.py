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
train = True ; test = False
#train = False; test = True

if train: record = tf.python_io.TFRecordWriter('train.tfrecords')
if test : record = tf.python_io.TFRecordWriter('test.tfrecords')
for index,[root, dirs, files] in enumerate(os.walk('.')):
    if index == 0:
        classes = dirs
        continue
    flag_files = int(len(files)*rate)
    for num,sample in enumerate(files):
        #227,227 is the size of Alexnet
        if train & (num > flag_files): continue
        if test  & (num <=flag_files): continue
        if train: print 'train: '+ str(index) + ':'+ str(num+1) + '/' + str(len(files))+ ':' + root + '/' +sample
        if test:  print 'test:  '+ str(index) + ':'+ str(num+1) + '/' + str(len(files))+ ':' + root + '/' +sample
        img = Image.open(root + '/' +sample)
        if img.mode != 'RGB':  img = img.convert('RGB')
        img = img.resize((227,227)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index - 1])),
            "img"  : tf.train.Feature(bytes_list=tf.train.BytesList(value=[  img    ]))
        }))
        record.write(example.SerializeToString())
    #if index == 2:
    #    break
record.close()
