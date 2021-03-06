import tensorflow as tf
from utils.load_tfrecords_64x128x3 import *
W  = '\033[0m'  # white (normal)
R  = '\033[31m' # red 
G  = '\033[32m' # green
O  = '\033[33m' # orange
#f1 = open("train.log","a")
import signal,sys
def signal_handler(signal, frame):
    try:
        print('You pressed Ctrl+C!Saving model...')
        saver.save(sess, 'model/'+__file__[:-3])
        print(str(step)+'Done!')
    except: 
        print('No model to save!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

X = tf.placeholder(tf.float32,shape=[None,64,128,3])
X_= tf.transpose(X, [0, 2, 1, 3])
Y = tf.placeholder(tf.float32,shape=[None])
Y_= tf.one_hot(tf.cast(Y,tf.int32),751)

#.conv1(11, 11, 96, 4, 4, padding='VALID')
#kernel_h=11;kernel_w=11;conv_out=96;strides_h=4;strides_w=4
w_conv1 = tf.Variable(tf.truncated_normal([11,11,3,15],stddev=0.01))
b_conv1 = tf.Variable(tf.constant(0.01,shape=[15]))
conv1   = tf.nn.relu(tf.nn.conv2d(X_, w_conv1, strides=[1,2,2,1],padding='VALID')+b_conv1)
#conv1.get_shape()=[None,55,55,96]

#.lrn(2, 2e-05, 0.75, name='norm1')
#radius=2;alpha=2e-05;beta=0.75;bias=1.0
norm1   = tf.nn.local_response_normalization(conv1,depth_radius=2,alpha=2e-05,beta=0.75,bias=1.0)
#norm1.get_shape()=[None,55,55,96]

#.max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#kernel_h=3;kernel_w=3;strides_h=2;strides_w=2
pool1   = tf.nn.max_pool(norm1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
#pool1.get_shape()=[None,27,27,96]

fc0_batch= tf.reshape(pool1,[-1,29,13*15])

x_biLSTM = tf.unpack(fc0_batch,axis=1)
cell_fw  = tf.nn.rnn_cell.BasicLSTMCell(195,forget_bias=1)
cell_bw  = tf.nn.rnn_cell.BasicLSTMCell(195,forget_bias=1)
biLSTM   = tf.nn.bidirectional_rnn(cell_fw, cell_bw, x_biLSTM, dtype=tf.float32, scope='1')

x_biLSTM2= biLSTM[0]
cell_fw2 = tf.nn.rnn_cell.BasicLSTMCell(390,forget_bias=1)
cell_bw2 = tf.nn.rnn_cell.BasicLSTMCell(390,forget_bias=1)
biLSTM2  = tf.nn.bidirectional_rnn(cell_fw2, cell_bw2, x_biLSTM2, dtype=tf.float32, scope='2')

x_biLSTM3= biLSTM2[0]
cell_fw3 = tf.nn.rnn_cell.BasicLSTMCell(780,forget_bias=1)
cell_bw3 = tf.nn.rnn_cell.BasicLSTMCell(780,forget_bias=1)
biLSTM3  = tf.nn.bidirectional_rnn(cell_fw3, cell_bw3, x_biLSTM3, dtype=tf.float32, scope='3')

x_biLSTM4= biLSTM3[0]
cell_fw4 = tf.nn.rnn_cell.BasicLSTMCell(1560,forget_bias=1)
cell_bw4 = tf.nn.rnn_cell.BasicLSTMCell(1560,forget_bias=1)
biLSTM4  = tf.nn.bidirectional_rnn(cell_fw4, cell_bw4, x_biLSTM4, dtype=tf.float32, scope='4')


#biLSTM[0][-1].shape = [?,56]
w_fc2    = tf.Variable(tf.random_normal([6240,751],stddev=0.01))
b_fc2    = tf.Variable(tf.constant(0.01,shape=[751]))
#y       = tf.matmul(biLSTM4[0][-1],w_fc2) + b_fc2
y        = tf.matmul(biLSTM5[0][-1],w_fc2) + b_fc2
y_       = tf.nn.softmax(y)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=Y_))    
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_,1),tf.argmax(Y_,1)),tf.float32))

tr_img, tr_label = load_tfrecords("market/train.tfrecords")
ts_img, ts_label = load_tfrecords("market/valid.tfrecords")
tr_img_batch, tr_label_batch = tf.train.shuffle_batch([tr_img, tr_label],batch_size=50,capacity=10802,min_after_dequeue=1000)
ts_img_batch, ts_label_batch = tf.train.shuffle_batch([ts_img, ts_label],batch_size=50,capacity=2132, min_after_dequeue=100)
#tr_num=10802, ts_num=2132

saver = tf.train.Saver()
sess  = tf.Session()
try:     sess.run(tf.global_variables_initializer())
except:  sess.run(tf.initialize_all_variables())
threads = tf.train.start_queue_runners(sess=sess)

#print 'loading ts_batch'
#ts_batch = sess.run([ts_img_batch, ts_label_batch])
#tr_batch = sess.run([tr_img_batch, tr_label_batch])
#print ts_batch[0].shape,ts_batch[1].shape
#print tr_batch[0].shape,tr_batch[1].shape
#print('Done!')
#raw_input('Done!')
##(-1,227x227,3)(-1)

for step in range(25000):
    tr_batch = sess.run([tr_img_batch, tr_label_batch])
    #tr_acc,loss,_ = sess.run([accuracy,cross_entropy,train_step],feed_dict={X:tr_batch[0],Y:tr_batch[1]})
    sess.run(train_step,feed_dict={X:tr_batch[0],Y:tr_batch[1]})
    if step%300 == 0: 
        tr_acc,loss = sess.run([accuracy,cross_entropy],feed_dict={X:tr_batch[0],Y:tr_batch[1]})
        ts_accs=[]
        for num_test in range(14):
            ts_batch = sess.run([ts_img_batch, ts_label_batch])
            ts_accs.append(sess.run(accuracy,feed_dict={X:ts_batch[0],Y:ts_batch[1]}))
        ts_acc = round(sum(ts_accs, 0.0) / len(ts_accs),4)
        s1 = R+str(step)+':loss:  '+str(loss  )+W
        s2 = O+str(step)+':tr_acc:'+str(tr_acc)+W
        s3 = G+str(step)+':ts_acc:'+str(ts_acc)+W
        print s1,s2,s3
        print [round(acc,3) for acc in ts_accs]        
        #f1.write(s1+s2+s3+'\n')
        #f1.flush()
saver.save(sess, 'model/'+__file__[:-3],global_step=step)
