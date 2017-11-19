# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:20:44 2017

@author: BadhreeshM
"""
"Load MNIST Data"
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data",one_hot=True, reshape=False)

"Start Tensorflow Session"
import tensorflow as tf
import matplotlib.pyplot as plt
import time
#sess = tf.Session()

"Input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch"
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
"Correct answers will go here"
Y_ = tf.placeholder(tf.float32, [None, 10])

"Weight and Bias Initialization"
def  weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

"2 convolutional layers with their channel count and a fully connected layer"
layer_1_depth = 64 # First convolutional layer output depth
layer_2_depth = 64 # Second convolutional layer output depth
FCL_size = 128 # No of units in fully connected layer

"Convolution and Pooling"
def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides= [1,2,2,1], padding='SAME')


"First Convolutional Layer, with 16 3x3 filters followed by max pooling" 
W_conv1 = weight_variable([3,3,1,layer_1_depth])
b_conv1 = bias_variable([layer_1_depth])

X_image = tf.reshape(X, [-1, 28, 28, 1])
#Convolve input image with the weight tensor, add bias and apply reLU
h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
#Now apply Max pooling to the feature map obtained above
h_pool1 = max_pool_2x2(h_conv1) # Image size reduced to 14x14


"Second Convolutional Layer, with 16 3x3 filters followed by max pooling"
W_conv2 = weight_variable([3,3,layer_1_depth,layer_2_depth])
b_conv2 = bias_variable([layer_2_depth])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # Image size reduced to 7x7

"Densely Connected layer: Since the image size has been reduced to 7x7, we add a fcl with 128 neurons to allow processing on the entire image"
W_fc1 = weight_variable([7 * 7 * layer_2_depth, FCL_size])
b_fc1 = bias_variable([FCL_size])
 
#Reshape the pooling layer tensor into a batch of vectors, multiple by wt matrix, add bias and apply reLU
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * layer_2_depth])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

"Final output layer of the network"
W_fc2 = weight_variable([FCL_size, 10])
b_fc2 = bias_variable([10])

Ylogits = tf.matmul(h_fc1, W_fc2) + b_fc2
Y = tf.nn.softmax(Ylogits)

"Training and Evaluating the Model"
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y_, logits = Ylogits ))
train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

t0 = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    validation_accuracy_list = []
    train_accuracy_list = []
    for i in range(33000):
        batch = mnist.train.next_batch(50)
        batch_val = mnist.validation.next_batch(50)
        
        "Occasionally report accuracy"
        
        if i % 1100 == 0:
            x = int(i/1100) + 1
            #train_accuracy = sess.run(accuracy, feed_dict={X: batch[0], Y_: batch[1]})
            validation_accuracy = sess.run(accuracy, feed_dict={X: batch_val[0], Y_: batch_val[1]})      
            print('Epoch %d, Validation Accuracy %g' % (x, validation_accuracy))
            #print('Epoch %d, Validation Accuracy %g' % (i, validation_accuracy))
            validation_accuracy_list.insert(x,validation_accuracy)
            #train_accuracy_list.insert(x,train_accuracy)
        
        "Run the training step"
        sess.run(train_step, feed_dict={X: batch[0], Y_: batch[1]})
    print('Test Accuracy: %g' % sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels }))
t1 = time.time()    
print('Duration: {:.1f} seconds'.format(t1-t0))

epochs =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
plt.plot(epochs, validation_accuracy_list)
plt.xlabel('No of Epochs')
plt.ylabel('Validation Accuracy')
plt.show()
 
        







 