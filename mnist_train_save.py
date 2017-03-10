#tensorflow mnist train

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tfhelpers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", help="start number for batch file")
parser.add_argument("-n", help="number of batches to read")
parser.add_argument("-d", help="device number for using in result file name")
args = parser.parse_args()
if args.n is not None:
    num_batches = int(args.n)
else:
    num_batches = 1

if args.s is not None:
    start_no = int(args.s)
else:
    start_no = 0

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

#sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(start_no, start_no + num_batches):
  #batch = mnist.train.next_batch(100)
  batch = tfhelpers.read_partition(None, i, "/tmp/")
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

save_path = saver.save(sess, "/tmp/model{0}.ckpt".format(dev_number))
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
