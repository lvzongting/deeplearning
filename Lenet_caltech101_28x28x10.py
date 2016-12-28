import tensorflow as tf
from utils.load_tfrecords_28x28x1 import *

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y0= tf.placeholder(tf.int32, shape=[None])
y_= tf.one_hot(y0,10)

convolv  = lambda x, w : tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
max_pool = lambda x    : tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#conv1 1,28,28,1 => 1,28,28,32
w_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]))
h_conv1 = tf.nn.relu(convolv(x,w_conv1) + b_conv1)
#pool1 1,28,28,32 => 1,14,14,32
h_pool1 = max_pool(h_conv1)
#conv2 1,28,28,32 => 1,14,14,64
w_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2 = tf.nn.relu(convolv(h_pool1,w_conv2) + b_conv2)
#pool2 1,14,14,64 => 1,7,7,64
h_pool2 = max_pool(h_conv2)
#densely 1,7,7,64 => 1024
w_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]))
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)
#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)
#Readout 1024 => 10
w_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]))
y     = tf.nn.softmax(tf.matmul(h_fc1_dropout,w_fc2)+b_fc2)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#for tr_num,_ in  enumerate(tf.python_io.tf_record_iterator("caltech101/train28x28.tfrecords")):pass
#for ts_num,_ in  enumerate(tf.python_io.tf_record_iterator("caltech101/test28x28.tfrecords" )):pass
tr_num = 1461; ts_num = 1445

tr_img, tr_label = load_tfrecords("caltech101/train28x28.tfrecords")
ts_img, ts_label = load_tfrecords("caltech101/test28x28.tfrecords" )
tr_img_batch, tr_label_batch = tf.train.shuffle_batch([tr_img, ts_label],batch_size=500,   capacity=2000,min_after_dequeue=1000)
ts_img_batch, ts_label_batch = tf.train.shuffle_batch([ts_img, ts_label],batch_size=ts_num,capacity=2000,min_after_dequeue=1000)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
threads = tf.train.start_queue_runners(sess=sess)

print 'loading ts_batch'
ts_batch = sess.run([ts_img_batch, ts_label_batch])
print ts_batch[0].shape

for i in range(20000):
     tr_batch = sess.run([tr_img_batch, tr_label_batch])
     #print batch[0].shape,one_hot(batch[1],num_labels=10)
     sess.run(train_step, feed_dict={x: tr_batch[0], y0: tr_batch[1], keep_prob: 0.5})
     print(str(i)+':'+str(sess.run(accuracy, feed_dict={x: ts_batch[0], y0: ts_batch[1], keep_prob: 1})))


