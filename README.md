# CNN_Architectures
This repository contains all the famous CNN networks and compares them on tiny imagenet dataset. 

### About Dataset
How many classes are there in ImageNet?
200 classes
It was created for students to practise their skills in creating models for image classification. The Tiny ImageNet dataset has 100,000 images across 200 classes. Each class has 500 training images, 50 validation images, and 50 test images.

## Arch 1 :- LeNet 
### About 
LeNet-5, from the paper <b>Gradient-Based Learning Applied to Document Recognition</b>, is a very efficient convolutional neural network for handwritten character recognition. <br>
Paper: Gradient-Based Learning Applied to Document Recognition <br>
Authors: <b>Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner</b> <br>
Published in: Proceedings of the IEEE (1998) <br> 

### Imp Points 
1) First ever implementation of backpropogation in Convolution Neural Network. 
2) Use of 5x5 Kernels with stride=1 in Convolution layer and 2x2 kernel with stride=2 in Average Pooling layer. <br> 
   Activation function used originally - Sigmoid 
3) A kernel used in convolution layer would extract features out of input images via the concept called edge detection. Earlier these kernels were hard coded but use of Neural Network allowed these kernels to be made of trainable weights that could adjust themselves by the concept of backpropogation. 
4) No.of kernel used in LeNet were 6 in 1st layer and 16 in 2nd layer. More the number of kernels (or filters) more amount of features extracted in a layer.
5) Architecture :- 
<img src="lenet-5.png" width="1000px" height="500px">

## Arch 2 :- AlexNet
### About 
The motivation behind AlexNet was going depper in convolution layers. Increase in no.of convolution layers meant extracting minor details in image aka extracting "features of features". But going deeper meant requirement of large no.of trainable parameters demanding more computation power. AlexNet proposed various measures to overcome these limitations in 2012 ImageNet Competition. <br> 
Author:<b> Alex Krizhevsky,Ilya Sutskever and Geoffrey Hinton</b> <br>

### Imp Points 
1) Large amount of filters used in conv layers ranging from 96 to 384.
2) 5 conv layers used with filter varying from 3x3 , 5x5 and 11x11. 
3) Earlier average pooling was being used which was a trainable layer. In AlexNet MaxPooling was introduced thereby making pooling layer a non-trainable layer reducing parameters to train. 
4) Several new small concepts were used for the first time in CNN which helped in increasing the depth of architecture like:
      --> Use of ReLu Activation function
      --> Use of Local Response Normalization becuase the output of ReLu is unbounded in nature.
      --> Concept of Dropout in Fully connected layers  to drop some neurons so as to prevent overfitting. ( Concept of deactivating neurons ) 
      
5) Use of Data augmentation- artificially increase the size of the training set-create a batch of "new" data from existing data by means of translation, flipping, noise.

6) Arch:-
<img src="alexnet2.png" width="1000px" height="500px">

## Arch 3 :- VGG
### About 
VGG is the Visual Geometry Group from Oxford University.The original purpose of VGG's research on the depth of convolutional networks is to understand how the depth of convolutional networks affects the accuracy of large-scale image classification and recognition. 
Paper: Very Deep Convolutional Networks for Large-Scale Image Recognition <br> 
Author:<b>K. Simonyan and A. Zisserman from the University of Oxford</b> <br>

### Imp Points 
1) A smaller 3 * 3 convolution kernel and a deeper network are used.The stack of two 3 * 3 convolution kernels is relative to the field of view of a 5 * 5 convolution kernel, and the stack of three 3 * 3 convolution kernels is equivalent to the field of view of a 7 * 7 convolution kernel. In this way, there can be fewer parameters (3 stacked 3 * 3 kernel have only 3*3*3=27 parameters whereas single 7 * 7 fitler has 49 paramters ( about 55% less in 3*3*3);on the other hand, they have more The non-linear transformation thereby increasing the ability of CNN to learn features.
2) One of the most symmetrical architecture in CNN. 3x3 Conv layers stacked on top of each other with increasing no.of filters in multiple of 2 starting from 64. After every two or three conv layers the output is maxpooled. 
3) Input is bit different in VGG: The input of VGG is set to an RGB image of 224x244 size. The average RGB value is calculated for all images on the training set,and then the image is given as input to VGG.
4) Although it is deeper and has more parameters than the AlexNet network, VGGNet can converge in less cycles for two reasons: one, the greater depth and smaller convolutions bring implicit regularization ; Second, some layers of pre-training.
5) The optimization method is a stochastic gradient descent SGD + momentum (0.9) with momentum. The batch size is 256.
6) Arch:-
<img src="vgg.png" width="1000px" height="500px">
