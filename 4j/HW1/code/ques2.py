import numpy.random as nr
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

arch = np.load('array_archive.npz')

a = arch['a']
b = arch['b']
c = [1]*500+[-1]*500
d = np.c_[np.r_[a,b],c]
nr.shuffle(d)
#print(d[:30])

iteration = 1000

train = d[:700]
test = d[700:]

X = tf.placeholder("float", [None, 2])  # create symbolic variables
Y = tf.placeholder("float", [None, 1])

W = tf.Variable(tf.random_normal(shape=[2,1]))
B =tf.Variable(tf.random_normal(shape=[1,1]))

model_output=tf.matmul(X,W)+B
loss = tf.reduce_mean(tf.square(Y-model_output))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # construct optimizer
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.sign(model_output), Y),tf.float32)) #construct accuracy
loss_array = []
acc_array = []
with tf.Session() as sess:
    with tf.device("/gpu:0"):
        tf.global_variables_initializer().run()
        for i in range(iteration):
            #print(i)
            _w = sess.run(W)
            _b = sess.run(B)
            sess.run(train_op, feed_dict={X: train[:,:2], Y: train[:,2:]})
            temp_loss = sess.run(loss, feed_dict={X: train[:,:2], Y: train[:,2:]})
            acc = sess.run(accuracy, feed_dict={X: test[:,:2], Y: test[:,2:]})
            print(temp_loss, acc)#, _w, _b)
            loss_array.append(temp_loss)
            acc_array.append(acc)


fig, axes = plt.subplots(1, 3)

axes[0].plot(a[:,0],a[:,1],'o',b[:,0],b[:,1],'o')
x = nr.uniform(-6,4,size=(1,100))
y = (-_w[0]/_w[1])*x - _b/_w[1]
axes[0].plot(x.T,y.T,'-')
axes[0].set_title('Final bound')

axes[1].plot(range(1,21),loss_array[:20])
axes[1].set_title('loss change')
axes[1].set_xlabel('iterations')
axes[1].set_ylabel('loss')

axes[2].plot(range(1,21),acc_array[:20])
axes[2].set_title('accuracy change')
axes[2].set_xlabel('iterations')
axes[2].set_ylabel('accuracy')

plt.show()







