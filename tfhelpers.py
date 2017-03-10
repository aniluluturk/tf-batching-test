#put partitioned data
import numpy, sys, os
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import random
mnist = None

def load_mnist(base_dir):
    global mnist
    mnist = input_data.read_data_sets("{0}tensorflow/mnist".format(base_dir), one_hot=True)

def create_partition(base_dir, batch_size=100, num_instances=1, randomize=False):
    global mnist
    filepaths = {"image": [], "label":[]}
    load_mnist(base_dir)
    pos_random = range(num_instances)
    if randomize:
        random.shuffle(pos_random)

    for i in range(num_instances):
        nb = mnist.train.next_batch(batch_size)
        pos = pos_random[i]
        image_path ="{0}mnist{1}.images".format(base_dir,pos)
        label_path="{0}mnist{1}.labels".format(base_dir,pos)
        imagef=open(image_path, "w+")
        labelf=open(label_path, "w+")
        filepaths["image"].append(image_path)
        filepaths["label"].append(label_path)
        numpy.save(imagef,nb[0])
        numpy.save(labelf,nb[1])
        imagef.close()
        labelf.close()
    print(filepaths)
    return filepaths

def create_test_files(base_dir):
    global mnist
    imagef = open("{0}mnist.test.images".format(base_dir), "w+")
    labelf = open("{0}mnist.test.labels".format(base_dir), "w+")
    numpy.save(imagef, mnist.test.images)
    numpy.save(labelf, mnist.test.labels)
    imagef.close()
    labelf.close()

def read_test_files(base_dir):
    global mnist
    imagef = open("{0}mnist.test.images".format(base_dir), "r")
    labelf = open("{0}mnist.test.labels".format(base_dir), "r")
    imagedata = numpy.load(imagef)
    labeldata = numpy.load(labelf)
    return imagedata, labeldata
    
def read_partition(filepaths, dev_num, base_dir=None):
    if filepaths is not None:
        imagef=open(filepaths.images[dev_num], "w+")
        labelf=open(filepaths.labels[dev_num], "w+")
    elif base_dir is not None:
        imagef=open("{0}mnist{1}.images".format(base_dir,dev_num), "r")
        labelf=open("{0}mnist{1}.labels".format(base_dir,dev_num), "r")
    imagef.seek(0)
    labelf.seek(0)
    #print imagef, type(imagef)
    imagearr = numpy.load("{0}mnist{1}.images".format(base_dir,dev_num))
    labelarr = numpy.load("{0}mnist{1}.labels".format(base_dir,dev_num))
    print "READ PARTITION {0}".format(dev_num)
    #print len(imagearr)
    #print len(labelarr)
    return imagearr,labelarr
    

def save_session(sess, path):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path) #preferably ending with .ckpt
    return save_path
    
def load_session(sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, path)
    
def load_and_evaluate(sess, path, x, y, y_, base_dir="/tmp/"):
    load_mnist(base_dir)
    load_session(sess, path)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    eval_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(eval_acc)
    return eval_acc
