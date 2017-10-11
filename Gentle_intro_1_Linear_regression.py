# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 15:29:55 2017

@author: BadhreeshM
"""

import tensorflow as tf
import numpy as np

''' First, we create the model'''
# Define Placeholders
x = tf.placeholder(tf.float32,[None,1]) # One feature(house size)
y_ = tf.placeholder(tf.float32,[None,1]) # One output(house price)

#Define Variables(Trainable parameters)
W = tf.Variable(tf.zeros([1,1])) # One output(house price) and one feature(house size)
b = tf.Variable(tf.zeros([1])) # One feature(house size)

#The linear Model
y = tf.matmul(x,W) + b

#min least squared cost function
cost = tf.reduce_mean(tf.square(y-y_))

    
#Implement gradient descent
learning_rate = 0.00001
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

'''Second, we train the model'''
#Variable initialization(All variables need to be initialized at the start of training)
init = tf.global_variables_initializer()

#Create a session(sess) and execute stuff using sess.run()
sess = tf.Session()
sess.run(init)

#Generate fake data
#We set the house price(ys) to always be 2 times the house size(xs) for simplicity
for i in range(100):
    xs = np.array([[i]])
    ys = np.array([[2*i]])

#Train
    feed = {x: xs,y_:ys}#Created a dictionary
#Similarly execute train_step in the loop by calling it within sess.run()
    sess.run(train_step,feed_dict=feed)
    
    print("After %d Iterations:"%i)
    print("W: %f"%sess.run(W))
    print("b: %f"%sess.run(b))
    








