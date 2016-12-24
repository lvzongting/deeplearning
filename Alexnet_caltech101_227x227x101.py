import tensorflow as tf
from utils.load_tfrecords import *
from numpy import *

net_data = load("model/bvlc_alexnet.npy").item()
#net_data.keys() 
#[u'fc6', u'fc7', u'fc8', u'conv3', u'conv2', u'conv1', u'conv5', u'conv4']
#net_data['conv1'][0].shape
#(11, 11, 3, 96)
#net_data['conv1'][1].shape
#(96,)

# (self.feed('data')
#          data=[-1,227,227,3]
#     .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#          kernel_h=11;kernel_w=11;conv_out=96;strides_h=4;strides_w=4
#     .lrn(2, 2e-05, 0.75, name='norm1')
#          radius=2;alpha=2e-05;beta=0.75;bias=1.0
#     .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#          kernel_h=3;kernel_w=3;strides_h=2;strides_w=2
#     .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#          kernel_h=5;kernel_w=5;conv_out=256;strids_h=1;strids_w=1
#     .lrn(2, 2e-05, 0.75, name='norm2')
#          radius=2;alpha=2e-05;beta=0.75;bias=1.0
#     .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#          kernel_h=3;kernel_w=3;strides_h=2;strides_w=2
#     .conv(3, 3, 384, 1, 1, name='conv3')
#          kernel_h=3;kernel_w=3;conv_out=384;strids_h=1;strids_w=1
#     .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#          kernel_h=3;kernel_w=3;conv_out=384;strids_h=1;strids_w=1
#     .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#          kernel_h=3;kernel_w=3;conv_out=256;strids_h=1;strids_w=1
#     .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
#          kernel_h=3;kernel_w=3;strides_h=2;strides_w=2;padding='VALID'
#     .fc(4096, name='fc6')
#     .fc(4096, name='fc7')
#     .fc(1000, relu=False, name='fc8')
#     .softmax(name='prob'))

x = tf.placeholder(tf.float32,shape=[None,227,227,3])
y0= tf.placeholder(tf.float32,shape=[None])
y_= tf.one_hot(tf.cast(y0,int32),101)
keep_prob = tf.placeholder(tf.float32)

#.conv1(11, 11, 96, 4, 4, padding='VALID')
#kernel_h=11;kernel_w=11;conv_out=96;strides_h=4;strides_w=4
#w_conv1 = tf.Variable(tf.truncated_normal([11,11,3,96],stddev=0.1))
#b_conv1 = tf.Variable(tf.constant(0.1,shape=[96]))
#w_conv1 = tf.Variable(net_data['conv1'][0])
#b_conv1 = tf.Variable(net_data['conv1'][1])
w_conv1 = tf.constant(net_data['conv1'][0],shape=net_data['conv1'][0].shape)
b_conv1 = tf.constant(net_data['conv1'][1],shape=net_data['conv1'][1].shape)
conv1   = tf.nn.relu(tf.nn.conv2d(x, w_conv1, strides=[1,4,4,1],padding='VALID')+b_conv1)
#conv1.get_shape()=[None,55,55,96]

#.lrn(2, 2e-05, 0.75, name='norm1')
#radius=2;alpha=2e-05;beta=0.75;bias=1.0
norm1   = tf.nn.local_response_normalization(conv1,depth_radius=2,alpha=2e-05,beta=0.75,bias=1.0)
#norm1.get_shape()=[None,55,55,96]

#.max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#kernel_h=3;kernel_w=3;strides_h=2;strides_w=2
pool1   = tf.nn.max_pool(norm1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
#pool1.get_shape()=[None,27,27,96]

#.conv(5, 5, 256, 1, 1, group=2, name='conv2')
#kernel_h=5;kernel_w=5;conv_out=256;strids_h=1;strids_w=1
#w_conv2 = tf.Variable(tf.truncated_normal([5,5,48,256],stddev=0.1))
#b_conv2 = tf.Variable(tf.constant(0.1,shape=[256]))
#w_conv2 = tf.Variable(net_data['conv2'][0])
#b_conv2 = tf.Variable(net_data['conv2'][1])
w_conv2 = tf.constant(net_data['conv2'][0],shape=net_data['conv2'][0].shape)
b_conv2 = tf.constant(net_data['conv2'][1],shape=net_data['conv2'][1].shape)
w_conv2g= tf.split(3,2,w_conv2)
pool1g  = tf.split(3,2,pool1)
conv2_1 = tf.nn.conv2d(pool1g[0],w_conv2g[0],strides=[1,1,1,1],padding='SAME')
conv2_2 = tf.nn.conv2d(pool1g[1],w_conv2g[1],strides=[1,1,1,1],padding='SAME')
conv2_a = tf.concat(3,[conv2_1,conv2_2])
conv2   = tf.nn.relu(conv2_a + b_conv2)
#conv2.get_shape()=[None,27,27,256]

#.lrn(2, 2e-05, 0.75, name='norm2')
#radius=2;alpha=2e-05;beta=0.75;bias=1.0
norm2   = tf.nn.local_response_normalization(conv2,depth_radius=2,alpha=2e-05,beta=0.75,bias=1.0)
#norm2.get_shape()=[None,27,27,256]

#.max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#kernel_h=3;kernel_w=3;strides_h=2;strides_w=2
pool2   = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
#pool2.get_shape()=[None,13,13,256]

#.conv(3, 3, 384, 1, 1, name='conv3')
#kernel_h=3;kernel_w=3;conv_out=384;strids_h=1;strids_w=1
#w_conv3 = tf.Variable(tf.truncated_normal([4,3,256,384],stddev=0.1))
#b_conv3 = tf.Variable(tf.constant(0.1,shape=[384]))
#w_conv3 = tf.Variable(net_data['conv3'][0])
#b_conv3 = tf.Variable(net_data['conv3'][1])
w_conv3 = tf.constant(net_data['conv3'][0],shape=net_data['conv3'][0].shape)
b_conv3 = tf.constant(net_data['conv3'][1],shape=net_data['conv3'][1].shape)
conv3   = tf.nn.relu(tf.nn.conv2d(pool2,w_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
#conv3.get_shape()=[None,13,13,384]

#.conv(3, 3, 384, 1, 1, group=2, name='conv4')
#kernel_h=3;kernel_w=3;conv_out=384;strids_h=1;strids_w=1
#w_conv4 = tf.Variable(tf.truncated_normal([3,3,192,384],stddev=0.1))
#b_conv4 = tf.Variable(tf.constant(0.1,shape=[384]))
#w_conv4 = tf.Variable(net_data['conv4'][0])
#b_conv4 = tf.Variable(net_data['conv4'][1])
w_conv4 = tf.constant(net_data['conv4'][0],shape=net_data['conv4'][0].shape)
b_conv4 = tf.constant(net_data['conv4'][1],shape=net_data['conv4'][1].shape)
w_conv4g= tf.split(3,2,w_conv4)
conv3g  = tf.split(3,2,conv3)
conv4_1 = tf.nn.conv2d(conv3g[0],w_conv4g[0],strides=[1,1,1,1],padding='SAME')
conv4_2 = tf.nn.conv2d(conv3g[1],w_conv4g[1],strides=[1,1,1,1],padding='SAME')
conv4_a = tf.concat(3,[conv4_1,conv4_2])
conv4   = tf.nn.relu(conv4_a + b_conv4)
#conv4.get_shape()=[None,13,13,384]

#.conv(3, 3, 256, 1, 1, group=2, name='conv5')
#kernel_h=3;kernel_w=3;conv_out=256;strids_h=1;strids_w=1
#w_conv5 = tf.Variable(tf.truncated_normal([3,3,192,256],stddev=0.1))
#b_conv5 = tf.Variable(tf.constant(0.1,shape=[256]))
#w_conv5 = tf.Variable(net_data['conv5'][0])
#b_conv5 = tf.Variable(net_data['conv5'][1])
w_conv5 = tf.constant(net_data['conv5'][0],shape=net_data['conv5'][0].shape)
b_conv5 = tf.constant(net_data['conv5'][1],shape=net_data['conv5'][1].shape)
w_conv5g= tf.split(3,2,w_conv5)
conv4g  = tf.split(3,2,conv4)
conv5_1 = tf.nn.conv2d(conv4g[0],w_conv5g[1],strides=[1,1,1,1],padding='SAME')
conv5_2 = tf.nn.conv2d(conv4g[0],w_conv5g[1],strides=[1,1,1,1],padding='SAME')
conv5_a = tf.concat(3,[conv5_1,conv5_2])
conv5   = tf.nn.relu(conv5_a + b_conv5)
#conv5.get_shape()=[None,13,13,256]

#.max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
#kernel_h=3;kernel_w=3;strides_h=2;strides_w=2;padding='VALID'
pool5   = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID') 
#pool5.get_shape()=[None,6,6,256]

#.fc(4096, name='fc6')
pool5_flat = tf.reshape(pool5,shape=[-1,6*6*256])
#w_fc6   = tf.Variable(tf.truncated_normal([6*6*256,4096],stddev=0.1))
#b_fc6   = tf.Variable(tf.constant(0.1,shape=[4096]))
#w_fc6   = tf.Variable(net_data['fc6'][0])
#b_fc6   = tf.Variable(net_data['fc6'][1])
w_fc6   = tf.constant(net_data['fc6'][0],shape=net_data['fc6'][0].shape)
b_fc6   = tf.constant(net_data['fc6'][1],shape=net_data['fc6'][1].shape)
fc6     = tf.nn.relu(tf.matmul(pool5_flat,w_fc6)+b_fc6)
fc6     = tf.nn.dropout(fc6,keep_prob)
#fc6.get_shape()=[None,4096]

#.fc(4096, name='fc7')
#w_fc7   = tf.Variable(tf.truncated_normal([4096,4096],stddev=0.1))
#b_fc7   = tf.Variable(tf.constant(0.1,shape=[4096]))
#w_fc7   = tf.Variable(net_data['fc7'][0])
#b_fc7   = tf.Variable(net_data['fc7'][1])
w_fc7   = tf.constant(net_data['fc7'][0],shape=net_data['fc7'][0].shape)
b_fc7   = tf.constant(net_data['fc7'][1],shape=net_data['fc7'][1].shape)
fc7     = tf.nn.relu(tf.matmul(fc6,w_fc7)+b_fc7)
#fc7     = tf.nn.dropout(fc7,keep_prob)
#fc7.get_shape()=[None,4096]

#.fc(1000, relu=False, name='fc8')
#w_fc8   = tf.Variable(tf.truncated_normal([4096,1000],stddev=0.1))
#b_fc8   = tf.Variable(tf.constant(0.1,shape=[1000]))
#w_fc8   = tf.Variable(net_data['fc8'][0])
#b_fc8   = tf.Variable(net_data['fc8'][1])
w_fc8   = tf.constant(net_data['fc8'][0],shape=net_data['fc8'][0].shape)
b_fc8   = tf.constant(net_data['fc8'][1],shape=net_data['fc8'][1].shape)
fc8     = tf.nn.relu(tf.matmul(fc7,w_fc8)+b_fc8)
#fc9     = tf.nn.dropout(fc9,keep_prob)

#.fc(101, relu=False, name='fc8')
w_fc9   = tf.Variable(tf.truncated_normal([1000,101],stddev=0.1))
b_fc9   = tf.Variable(tf.constant(0.1,shape=[101]))
y       = tf.nn.softmax(tf.matmul(fc8,w_fc9)+b_fc9)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step    = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))

#for tr_num,_ in  enumerate(tf.python_io.tf_record_iterator("caltech101/train.tfrecords")):pass
#for ts_num,_ in  enumerate(tf.python_io.tf_record_iterator("caltech101/test.tfrecords" )):pass
tr_num = 4410; ts_num = 4265

tr_img, tr_label = load_tfrecords("caltech101/train.tfrecords")
ts_img, ts_label = load_tfrecords("caltech101/test.tfrecords" )
tr_img_batch, tr_label_batch = tf.train.shuffle_batch([tr_img, ts_label],batch_size=300,capacity=2000,min_after_dequeue=1000)
ts_img_batch, ts_label_batch = tf.train.shuffle_batch([ts_img, ts_label],batch_size=200,capacity=2000,min_after_dequeue=1000)

sess = tf.Session()
try:     sess.run(tf.global_variables_initializer())
except:  sess.run(tf.initialize_all_variables())
threads = tf.train.start_queue_runners(sess=sess)

print 'loading ts_batch'
ts_batch = sess.run([ts_img_batch, ts_label_batch])
print ts_batch[0].shape,ts_batch[1].shape
#raw_input("Break 1")

for i in range(20000):
     tr_batch = sess.run([tr_img_batch, tr_label_batch])
     #print batch[0].shape,one_hot(batch[1],num_labels=10)
     sess.run(train_step, feed_dict={x: tr_batch[0], y0: tr_batch[1], keep_prob: 1})
     print(str(i)+':'+str(sess.run(accuracy, feed_dict={x: tr_batch[0], y0: tr_batch[1], keep_prob: 1})))
     if i%50 == 0:   print(str(i)+':'+str(sess.run(accuracy, feed_dict={x: ts_batch[0], y0: ts_batch[1], keep_prob: 1})))


