import tensorflow as tf
from utils.load_tfrecords_227x227x3 import *
from numpy import *
W  = '\033[0m'  # white (normal)
G  = '\033[32m' # green
O  = '\033[33m' # orange


X = tf.placeholder(tf.float32,shape=[None,227,227,3])
Y = tf.placeholder(tf.float32,shape=[None])
Y_= tf.one_hot(tf.cast(Y,int32),101)

x        = tf.reshape(X,[-1,227,227*3])
x_fc0    = tf.reshape(x,[-1,227*3])
w_fc0    = tf.Variable(tf.random_normal([227*3,64],stddev=0.01))
b_fc0    = tf.Variable(tf.constant(0.01,shape=[64]))
fc0      = tf.matmul(x_fc0,w_fc0)+b_fc0

fc0_batch= tf.reshape(fc0,[-1,227,64])

cell_fw  = tf.nn.rnn_cell.BasicLSTMCell(64,forget_bias=1)
cell_bw  = tf.nn.rnn_cell.BasicLSTMCell(64,forget_bias=1)

x_biLSTM = tf.unpack(fc0_batch,axis=1)
biLSTM   = tf.nn.bidirectional_rnn(cell_fw, cell_bw, x_biLSTM, dtype=tf.float32)


#biLSTM[0][-1].shape = [?,56]
w_fc2    = tf.Variable(tf.random_normal([128,101],stddev=0.01))
b_fc2    = tf.Variable(tf.constant(0.01,shape=[101]))
y        = tf.matmul(biLSTM[0][-1],w_fc2) + b_fc2
y_       = tf.nn.softmax(y)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=Y_))    
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1),tf.argmax(Y_,1)),tf.float32))

tr_img, tr_label = load_tfrecords("caltech101/train.tfrecords")
ts_img, ts_label = load_tfrecords("caltech101/test.tfrecords" )
tr_img_batch, tr_label_batch = tf.train.shuffle_batch([tr_img, ts_label],batch_size=6000,capacity=7500,min_after_dequeue=1000)
ts_img_batch, ts_label_batch = tf.train.shuffle_batch([ts_img, ts_label],batch_size=600,capacity=750,min_after_dequeue=100)
#tr_num=7520, ts_num=774

sess = tf.Session()
try:     sess.run(tf.global_variables_initializer())
except:  sess.run(tf.initialize_all_variables())
threads = tf.train.start_queue_runners(sess=sess)

print 'loading ts_batch'
ts_batch = sess.run([ts_img_batch, ts_label_batch])
tr_batch = sess.run([tr_img_batch, tr_label_batch])
print ts_batch[0].shape,ts_batch[1].shape
print tr_batch[0].shape,tr_batch[1].shape
print('Done!')
#raw_input('Done!')
#(-1,227x227,3)(-1)

for i in range(20000):
    tr_acc,loss,_ = sess.run([accuracy,cross_entropy,train_step],feed_dict={X:tr_batch[0],Y:tr_batch[1]})
    print(O+str(i)+':loss:  '+str(loss  )+W)
    print(W+str(i)+':tr_acc:'+str(tr_acc)+W)
    ts_acc        = sess.run(accuracy,feed_dict={X:ts_batch[0],Y:ts_batch[1]})
    print(G+str(i)+':ts_acc:'+str(ts_acc)+W)

