import tensorflow as tf
import numpy as np
from pytz.tzinfo import _epoch
tf.reset_default_graph()
lr = 0.01
x = np.arange(-1, 100, 0.2)
y = x*2.3 + 6 + np.random.normal(scale = 0.2)
X = tf.placeholder(float64, x.shape)
Y = tf.placeholder(float64, y.shape)
batch_size = 64
W = tf.get_variable("weight", [1,1], float32)
B = tf.get_variable("bias", [1,1], float32)
def forward_prop(x, w, b):
    return tf.matmul(x, w) + b

def calculate_cost(y, x, w, b):
    return tf.losses.mean_squared_error(y, forward_prop(x, w, b))
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _epoch in range(5):
        shuffled_indices = np.random.shuffle(range(len(x)))
        begin_ind = 0
        end_ind = begin_ind + batch_size
        while end_ind < len(shuffled_indices):
            begin_ind = end_ind
            end_ind = min(len(shuffled_indices), end_ind + batch_size) 

            
    cost = sess.run(X,Y,feed_dict = {X: })

