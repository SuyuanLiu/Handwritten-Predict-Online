import tensorflow as tf 
import input_data
import model 
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

mnist = input_data.read_data_sets('mnist_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
y = model.cnn(x_image, keep_prob)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./checkpoints', sess.graph)
    summary_writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i % 100 == 0:
            train_acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print(("step %d, accuracy %g")%(i, train_acc))
            path = saver.save(sess, './checkpoints/cnn/cnn.ckpt')
        
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
    path = saver.save(sess, './checkpoints/cnn/cnn.ckpt')
    print('saver saved')
        
