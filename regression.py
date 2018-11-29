import tensorflow as tf 
import input_data
import model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = model.regression(x)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
        print(i, sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

        path = saver.save(sess, './checkpoints/regression/regression.ckpt')
        print('model saved')