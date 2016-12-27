# 深度学习与tensorflow学习笔记
## 一个简单的alexnet     
+ 主页[http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/] 
+ 代码[http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/myalexnet.py]
+ 模型参数 bvlc_alexnet.npy[https://drive.google.com/open?id=0B7Wy478uBsx8eTI4OTVOeGxtcVE]

## RNN & lstm 网络
+ 手写数字识别[lstm_mnist_28x28x10.py]
+ Caltech101分类-灰度[lstm_caltech_227x227x101L.py]
+ 图片生成 -- DRAW: A Recurrent Neural Network For Image Generation by Gregor et al. (Review by Tim Cooijmans)[https://github.com/tensorflow/magenta/blob/master/magenta/reviews/draw.md]
+ 序列生成 -- Generating Sequences with Recurrent Neural Networks by Graves. (Review by David Ha)[https://github.com/tensorflow/magenta/blob/master/magenta/reviews/summary_generation_sequences.md]

## Pixel RNN
+ 笔记 Pixel Recurrent Neural Networks by Van den Oord et al. (Review by Kyle Kastner)[https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md]
+ 代码 [https://github.com/carpedm20/pixel-rnn-tensorflow]
+ 论文 [https://arxiv.org/abs/1601.06759] 

##Generative Adversarial Networks 网络
+ Ian Goodfellow--2016talk [https://www.youtube.com/watch?v=HN9NRhm9waY]
+ Ian Goodfellow--Review by Max Strakhov [https://github.com/tensorflow/magenta/blob/master/magenta/reviews/GAN.md]
+ Improved Techniques for Training GANs [https://arxiv.org/abs/1606.03498]

##Deep Belief Networks
+ 文档[http://deeplearning.net/tutorial/DBN.html]

##Deep Residual Networks
+ He Kaiming----MSRA 2015 [https://github.com/KaimingHe/deep-residual-networks]
+ He Kaiming----CVPR 2016 [https://www.youtube.com/watch?v=C6tLw-rPQ2o]

# 数据库 Dataset as tfrecords form
##Caltech101
+ files: train28x28.tfrecords test28x28.tfrecords train.tfrecords test.tfrecords
+ 生成脚本 [utils/save_dirs_to_tfrecords.py] [utils/save_dirs_to_tfrecords_28x28.py]
+ google drive[https://drive.google.com/open?id=0B7Wy478uBsx8YWpLWEQyLVVQRjA]
