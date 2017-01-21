import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

#X.shape = [-1,784]

X = tf.placeholder(tf.float32,shape=[None,784])
Y = tf.placeholder(tf.float32,shape=[None,10])

x = tf.reshape(X,[-1,28,28])


cell_fw  = tf.nn.rnn_cell.BasicLSTMCell(28,forget_bias=0.1)
cell_bw  = tf.nn.rnn_cell.BasicLSTMCell(28,forget_bias=0.1)

x_biLSTM = tf.unpack(x,axis=1)
biLSTM   = tf.nn.bidirectional_rnn(cell_fw, cell_bw, x_biLSTM, dtype=tf.float32)

#biLSTM[0][-1].shape = [?,56]
w_fc2    = tf.Variable(tf.random_normal([56,10]))
b_fc2    = tf.Variable(tf.random_normal([10]))
fc2      = tf.matmul(biLSTM[0][-1],w_fc2) + b_fc2

y        = tf.nn.softmax(fc2)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2,labels=Y)    
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(Y,1)),tf.float32))

batch = mnist.train.next_batch(500)
print batch[0].shape,batch[1].shape
#(500,784)(500,10)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(2000):
    batch = mnist.train.next_batch(500)
    sess.run(train_step,feed_dict={X:batch[0],Y:batch[1]})
    #print(str(i)+':'+str(sess.run(accuracy,feed_dict={X:batch[0],Y:batch[1]})))
    print(str(i)+':'+str(sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels})))



    
