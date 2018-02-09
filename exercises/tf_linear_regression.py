from random import shuffle
import tensorflow as tf
import numpy as np
from pytz.tzinfo import _epoch
tf.reset_default_graph()
lr = 0.01
x = np.arange(-1, 100, 0.01)
y = x*2.3 + 10 + np.random.normal(scale = 0.2)
batch_size = 1024
X = tf.placeholder(tf.float64, [batch_size,1])
Y = tf.placeholder(tf.float64, [batch_size,1])
W = tf.get_variable("weight", (1,1), tf.float64, initializer = tf.random_normal_initializer())
B = tf.get_variable("bias", (1,1), tf.float64, initializer = tf.constant_initializer(0))
def forward_prop(x, w, b):
    return tf.matmul(x, w) + b

def calculate_cost(y, x, w, b):
    return tf.losses.mean_squared_error(y, forward_prop(x, w, b))

optimizer = tf.train.AdamOptimizer(learning_rate = .6).minimize(calculate_cost(Y, X, W, B))

    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _epoch in range(50):
        shuffled_indices = list(range(len(x)))
        shuffle(shuffled_indices)
        begin_ind = 0
        end_ind = begin_ind + batch_size
        while begin_ind < len(shuffled_indices):
            sess.run(optimizer, feed_dict = {X: x[min(begin_ind, end_ind - batch_size):end_ind,None], Y: y[min(begin_ind, end_ind - batch_size):end_ind,None]})
            cost = sess.run(calculate_cost(Y, X, W, B), feed_dict = {X: x[min(begin_ind, end_ind - batch_size):end_ind,None], Y: y[min(begin_ind, end_ind - batch_size):end_ind,None]})
            weight, bias = sess.run([W,B])
            print(weight)
            print(bias)
            begin_ind = end_ind
            end_ind = min(len(shuffled_indices), end_ind + batch_size) 
    weight, bias = sess.run([W,B])
    print(weight)
    print(bias)