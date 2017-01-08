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

## Matching Networks for One Shot Learning
+ 论文 Matching Networks for One Shot Learning [https://arxiv.org/abs/1606.04080]
+ 笔记 [https://github.com/karpathy/paper-notes/blob/master/matching_networks.md]
+ 代码 [https://github.com/zergylord/oneshot]


##Generative Adversarial Networks 网络
+ Ian Goodfellow--2016talk [https://www.youtube.com/watch?v=HN9NRhm9waY]
+ Ian Goodfellow--Review by Max Strakhov [https://github.com/tensorflow/magenta/blob/master/magenta/reviews/GAN.md]
+ Improved Techniques for Training GANs [https://arxiv.org/abs/1606.03498]
+ 各种GAN改进的概览--CGAN-LAPGAN-DCGAN-GRAN-VAEGAN-- [http://chuansong.me/n/317902651864] 

## DCGAN 生成网络
+ Image Completion with Deep Learning in TensorFlow ---- Brandon Amos [https://bamos.github.io/2016/08/09/deep-completion/]
+ Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks[https://arxiv.org/abs/1511.06434]
+ DCGAN - How does it work? [https://www.youtube.com/watch?v=VAeEt9df-hQ]
+ DCGAN with SVHN dataset github code [https://github.com/shuyo/iir/blob/master/dnn/dcgan-svhn.py]
+ 代码 [dcgan_svhn_32x32x3.py]
+ 结果 [https://drive.google.com/open?id=0B7Wy478uBsx8ekhVbEVqeXU5cTA]
+ 优化DCGAN网络 论文:Improved Techniques for Training GANs [https://arxiv.org/abs/1606.03498] 代码 [https://github.com/openai/improved-gan]  \ppt


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

# Topic
##生成艺术品和音乐 Magenta
+ 主页 [https://github.com/tensorflow/magenta]
+ 相关论文 [https://github.com/tensorflow/magenta/tree/master/magenta/reviews]

