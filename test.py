import numpy as np 
import tensorflow as tf 
from flask import Flask, jsonify, render_template, request 
import model 


class Predict:
    def __init__(self):
        self.graph = tf.Graph()   # 为每个类（实例）单独创建一个graph
        with self.graph.as_default():
            self.saver = 


x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
keep_prob = tf.placeholder(tf.float32)

sess = tf.Session()

y1 = model.regression(x)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('checkpoints/regression/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

y2 = model.cnn(x_image, keep_prob)
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('checkpoints/cnn/')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

def regression(input):
    return sess.run(y1, feed_dict={x:input}).flatten().tolist()
def cnn(input):
    return sess.run(y2, feed_dict={x:input, keep_prob:1.0}).flatten().tolist()




app = Flask(__name__)

@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) /255.0).reshape([1, 784])
    output1 = regression(input)
    output2 = cnn(input)
    return jsonify(results=[output1, output2])

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8000)



















# import numpy as np 
# import tensorflow as tf 
# from flask import Flask, jsonify, render_template, request 
# import model 

# x = tf.placeholder(tf.float32, [None, 784])
# x_image = tf.reshape(x, [-1, 28, 28, 1])
# keep_prob = tf.placeholder(tf.float32)
# y = model.cnn(x_image, keep_prob)

# saver = tf.train.Saver()
# with tf.Session() as sess:
#     ckpt = tf.train.get_checkpoint_state('checkpoints/cnn/')
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#         print('yes')
#     else:
#         print('no')

#     inn = np.zeros([2, 784])
#     y_pre = sess.run(y, feed_dict={x:inn, keep_prob:1.0})
#     print(y_pre)

