import tensorflow as tf
from utils.load_tfrecords3 import *


#x.shape = [-1,227,227]
#y.shape = [-1,101]

x = tf.placeholder(tf.float32,[None, 227, 227])
y0= tf.placeholder(tf.float32,[None])
y_= tf.one_hot(tf.cast(y0,tf.int32),101)

#time_major=False x = [bacth_size, max_time, input_data]
#input hidden layer [bacth_size,max_time,input_data[227]]=>[bacth_size,max_time,input_data[1024]]
#x_input = [bacth_size*max_time,input_data]
x_hidden1 = tf.reshape(x,[-1,227])
w_hidden1 = tf.Variable(tf.truncated_normal([227,1024],stddev=0.1))
b_hidden1 = tf.Variable(tf.constant(0.1,shape=[1024]))
hidden1   = tf.matmul(x_hidden1, w_hidden1)+ b_hidden1

#lstm size=1024
#time_major=False x = [bacth_size, max_time, input_data]
lstm1_input = tf.reshape(hidden1,[-1,227,1024])
lstm1_cell  = tf.nn.rnn_cell.BasicLSTMCell(1024,forget_bias=1.0,state_is_tuple=True)
lstm1_init_state = lstm1_cell.zero_state(batch_size=500,dtype=tf.float32)
lstm1_outputs, lstm1_state = tf.nn.dynamic_rnn(lstm1_cell,lstm1_input,initial_state=lstm1_init_state, time_major=False)
#lstm_outputs.get_shape = [1000,227,1024]

x_hidden2 = tf.unpack(tf.transpose(lstm1_outputs,[1,0,2]))
w_hidden2 = tf.Variable(tf.truncated_normal([1024,101],stddev=0.1))
b_hidden2 = tf.Variable(tf.constant(0.1,shape=[101]))
hidden2   = tf.matmul(x_hidden2[-1],w_hidden2)+b_hidden2

y = tf.nn.softmax(hidden2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step    = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))

#for all_num,_ in enumerate(tf.python_io.tf_record_iterator('caltech101/all_L.tfrecords')):pass
#all_num = 

tr_img, tr_label = load_tfrecords('caltech101/train_L.tfrecords')
ts_img, ts_label = load_tfrecords('caltech101/test_L.tfrecords')
tr_img_batch, tr_label_batch = tf.train.shuffle_batch([tr_img, tr_label], batch_size=500, capacity=3000, min_after_dequeue=10)
ts_img_batch, ts_label_batch = tf.train.shuffle_batch([ts_img, ts_label], batch_size=500, capacity=3000, min_after_dequeue=10)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
threads = tf.train.start_queue_runners(sess=sess)
print 'loading ts_batch...'
ts_batch = sess.run([ts_img_batch, ts_label_batch])
print ts_batch[0].shape,ts_batch[1].shape

for i in range(20000):
    tr_batch = sess.run([tr_img_batch, tr_label_batch])
    sess.run(train_step, feed_dict={x: tr_batch[0],y0:tr_batch[1]}) 
    print('tr:'+str(i)+':'+str(sess.run(accuracy,feed_dict={x: tr_batch[0], y0: tr_batch[1]})))
    print('ts:'+str(i)+':'+str(sess.run(accuracy,feed_dict={x: ts_batch[0], y0: ts_batch[1]})))



