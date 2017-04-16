
# This was one of my first interactions with tensorflow.
# You can find quite similar implementation in the official docs of Tensorflow docs.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

# weights and biases for the first layer
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# predicteed value of y
y = tf.nn.softmax(tf.matmul(x, W) + b)


# actual value of y
y_ = tf.placeholder(tf.float32, [None, 10])


# Determining the loss of our model by using cross entropy
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y) )

# Training our model with backprop and gradient descent optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# Initialize our variables
init = tf.global_variables_initializer()


# Calculating how many correct predictions
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# converting bool array to float and calculating accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Start our session
with tf.Session() as sess:
	sess.run(init)

	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
		if i % 100 == 0:
			print("Batch Number {}: Accuracy {}".format(i, 100*sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))

	print("Final Accuracy : ", end="")
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))



