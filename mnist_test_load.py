#mnist.load

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tfhelpers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", help="device no")
args = parser.parse_args()

if args.d is not None:
    dev_number = args.d
else:
    dev_number = ""


sess = tf.InteractiveSession()
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()
sess.run(init_op)

y = tf.nn.softmax(tf.matmul(x,W) + b)

#saver.restore(sess, "/tmp/model{0}.ckpt".format(dev_number))
#print("Model restored.")
tfhelpers.load_and_evaluate(sess, "/tmp/model{0}.ckpt".format(dev_number), x, y, y_)
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#images, labels = tfhelpers.read_test_files("/tmp/")
#print(accuracy.eval(feed_dict={x: images, y_: labels}))
