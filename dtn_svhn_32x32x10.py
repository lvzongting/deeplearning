import tensorflow as tf
from utils.load_tfrecords_32x32x3 import *

#x = [-1,32,32,3] y = [-1,1]
x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_= tf.placeholder(tf.float32,shape=[None,10])
#y_= tf.reshape(tf.one_hot(tf.cast(y0,tf.int32),10),[-1,10])

#conv1 64
w_conv1 = tf.Variable(tf.truncated_normal([5,5,3,64],stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1,shape=[64]))
conv1   = tf.nn.relu(tf.nn.conv2d(x,w_conv1,strides=[1,1,1,1],padding='SAME')+b_conv1)
#conv1.get_shape() =[None,32,32,64] 
#pool1
pool1   = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#pool1.get_shape() =[None,16,16,64]
#conv2 128
w_conv2 = tf.Variable(tf.truncated_normal([5,5,64,128],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1,shape=[128]))
conv2   = tf.nn.relu(tf.nn.conv2d(pool1,w_conv2,strides=[1,1,1,1],padding='SAME')+b_conv2)
#conv2.get_shape() =[None,16,16,128] 
#pool2
pool2   = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#pool2.get_shape() =[None,8,8,128]
#conv3 256
w_conv3 = tf.Variable(tf.truncated_normal([5,5,128,256],stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1,shape=[256]))
conv3   = tf.nn.relu(tf.nn.conv2d(pool2,w_conv3,strides=[1,1,1,1],padding='SAME')+b_conv3)
#conv3.get_shape() =[None,8,8,256]
#pool3
pool3   = tf.nn.max_pool(conv3,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
#pool3.get_shape() =[None,4,4,256]
#conv4 128
w_conv4  = tf.Variable(tf.truncated_normal([2,2,256,128],stddev=0.1))
b_conv4  = tf.Variable(tf.constant(0.1,shape=[128]))
conv4    = tf.nn.relu(tf.nn.conv2d(pool3,w_conv4,strides=[1,1,1,1],padding='SAME')+b_conv4)
#conv4.get_shape() =[None,4,4,128]
#pool4
pool4    = tf.nn.max_pool(conv4,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
#pool4.get_shape() =[None,1,1,128]
#fc1 10
pool4_flat = tf.reshape(pool4,[-1,128])
w_fc1    = tf.Variable(tf.truncated_normal([128,10],stddev=0.1))
b_fc1    = tf.Variable(tf.constant(0.1,shape=[10]))
#fc1     = tf.nn.relu(tf.matmul(pool4,w_fc1)+b_fc1)
y        = tf.nn.softmax(tf.matmul(pool4_flat,w_fc1)+b_fc1)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step    = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))

tr_img, tr_label = load_tfrecords('svhn/train.tfrecords')
ts_img, ts_label = load_tfrecords('svhn/test.tfrecords')
tr_img_batch, tr_label_batch = tf.train.shuffle_batch([tr_img, tr_label], batch_size=2000, capacity=3000, min_after_dequeue=10)
ts_img_batch, ts_label_batch = tf.train.shuffle_batch([ts_img, ts_label], batch_size=2600, capacity=3000, min_after_dequeue=10)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
threads = tf.train.start_queue_runners(sess=sess)
print 'loading ts_batch...'
ts_batch = sess.run([ts_img_batch, ts_label_batch])
#tr_batch = sess.run([tr_img_batch, tr_label_batch])
print ts_batch[0].shape,ts_batch[1]
#raw_input('Done')

for i in range(20000):
    tr_batch = sess.run([tr_img_batch, tr_label_batch])
    sess.run(train_step,feed_dict={x: tr_batch[0],y_: tr_batch[1]})
    print('tr:'+str(i)+' '+str(sess.run(accuracy,feed_dict={x: tr_batch[0],y_: tr_batch[1]})))
    print('ts:'+str(i)+' '+str(sess.run(accuracy,feed_dict={x: ts_batch[0],y_: ts_batch[1]})))
