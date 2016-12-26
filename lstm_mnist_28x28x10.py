import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data',one_hot=True)

#x.shape = [-1,784]
#y.shape = [-1,10]

x = tf.placeholder(tf.float32,[None, 784])
y_= tf.placeholder(tf.float32,[None, 10])

#time_major=False x = [bacth_size, max_time, input_data]
#input hidden layer [bacth_size,max_time,input_data[28]]=>[bacth_size,max_time,input_data[128]]
#x_input = [bacth_size*max_time,input_data]
x_hidden1 = tf.reshape(x,[-1,28])
w_hidden1 = tf.Variable(tf.truncated_normal([28,128],stddev=0.1))
b_hidden1 = tf.Variable(tf.constant(0.1,shape=[128]))
hidden1   = tf.matmul(x_hidden1, w_hidden1)+ b_hidden1

#lstm size=128
#time_major=False x = [bacth_size, max_time, input_data]
lstm1_input = tf.reshape(hidden1,[-1,28,128])
lstm1_cell  = tf.nn.rnn_cell.BasicLSTMCell(128,forget_bias=1.0,state_is_tuple=True)
lstm1_init_state = lstm1_cell.zero_state(batch_size=128,dtype=tf.float32)
lstm1_outputs, lstm1_state = tf.nn.dynamic_rnn(lstm1_cell,lstm1_input,initial_state=lstm1_init_state, time_major=False)
#lstm_outputs.get_shape = [128,28,128]

x_hidden2 = tf.unpack(tf.transpose(lstm1_outputs,[1,0,2]))
w_hidden2 = tf.Variable(tf.truncated_normal([128,10],stddev=0.1))
b_hidden2 = tf.Variable(tf.constant(0.1,shape=[10]))
hidden2   = tf.matmul(x_hidden2[-1],w_hidden2)+b_hidden2

y = tf.nn.softmax(hidden2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step    = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
accuracy      = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
test_batch_xs, test_batch_ys = mnist_data.train.next_batch(128)
for i in range(2000):
    batch_xs, batch_ys = mnist_data.train.next_batch(128)
    sess.run(train_step, feed_dict={x: batch_xs,y_:batch_ys}) 
    print(str(i)+':'+str(sess.run(accuracy,feed_dict={x: test_batch_xs, y_: test_batch_ys})))



