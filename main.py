import numpy as np 
import tensorflow as tf 
from flask import Flask, jsonify, render_template, request 

import model 

# x = tf.placeholder(tf.float32, [None, 784])
# sess = tf.Session()

# with tf.variable_scope('regression'):
#     y1 = model.regression(x)
#     # saver = tf.train.Saver()
#     saver = tf.train.import_meta_graph('checkpoints/regression/regression.ckpt.meta')
#     tf.train.latest_checkpoint('checkpoints/regression/regression.ckpt')
#     # saver.restore(sess, "checkpoints/regression/regression.ckpt")

# with tf.variable_scope('cnn'):
#     keep_prob = tf.placeholder(tf.float32)
#     x_image = tf.reshape(x, [-1, 28, 28, 1])
#     y2 = model.cnn(x_image, keep_prob)
#     # saver = tf.train.Saver()
#     saver2 = tf.train.import_meta_graph('checkpoints/cnn/cnn.ckpt.meta')
#     tf.train.latest_checkpoint('checkpoints/cnn/cnn.ckpt')
    # saver.restore(sess, "checkpoints/cnn/cnn.ckpt")

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()

# with tf.variable_scope('regression'):
y1 = model.regression(x)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('checkpoints/regression/')
# if ckpt and ckpt.model_checkpoint_path:
saver.restore(sess, ckpt.model_checkpoint_path)

# # with tf.variable_scope('cnn'):
# y2 = model.cnn(x_image, keep_prob)
# saver = tf.train.Saver()
# ckpt = tf.train.get_checkpoint_state('checkpoints/cnn/')
# # if ckpt and ckpt.model_checkpoint_path:
# saver.restore(sess, ckpt.model_checkpoint_path)

def regression(input):
    return sess.run(y1, feed_dict={x:input}).flatten().tolist()
def cnn(input):
    return sess.run(y2, feed_dict={x:input, keep_prob:1.0}).flatten().tolist()

app = Flask(__name__)

@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) /255.0).reshape([1, 784])
    output1 = regression(input)
    output2 = output1
    # output2 = cnn(input)
    # output1 = output2
    return jsonify(results=[output1, output2])

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8000)