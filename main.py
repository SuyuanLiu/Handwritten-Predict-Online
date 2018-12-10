import numpy as np 
import tensorflow as tf 
from flask import Flask, jsonify, render_template, request 
import model 


class Predict:
    def __init__(self, model_name, checkpoint_dir):
        self.graph = tf.Graph()   # create graph for each instance individuly
        self.model_name = model_name

        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [None, 784])
            self.keep_prob = tf.placeholder(tf.float32)
            
            if self.model_name == 'regression':
                self.output = model.regression(self.x)
            else:
                self.output = model.cnn(self.x, self.keep_prob)
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        
    def predict(self, input):
        if self.model_name == 'regression':
            return self.sess.run(self.output, feed_dict={self.x:input}).flatten().tolist()
        else:
            return self.sess.run(self.output, feed_dict={self.x:input, self.keep_prob:1.0}).flatten().tolist()
            
def regression(input):
    regression_pre = Predict('regression', 'checkpoints/regression/')
    return regression_pre.predict(input)
def cnn(input):
    cnn_pre = Predict('cnn', 'checkpoints/cnn/')
    return cnn_pre.predict(input)

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
