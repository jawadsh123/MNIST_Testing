import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("../MNIST_data", one_hot=True)

# universal constants
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10

# creating a class attr for data
data.test.cls = np.array([label.argmax() for label in data.test.labels])


# ##############################################
# creating tensorflow graph
# ##############################################

# x features for our graph (in this case images of numbers)
x = tf.placeholder(tf.float32, [None, img_size_flat])

# placeholder for true y value
y_true = tf.placeholder(tf.float32, [None, num_classes])

# placeholder for true classes
y_true_cls = tf.placeholder(tf.int64, [None])

# defining weights and biases
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# output logits (output after applying weights and biases)
logits = tf.matmul(x, weights) + biases

# applying softmax to the logits to get the classes
y_pred = tf.nn.softmax(logits)

# getting predicted class from y_pred
y_pred_cls = tf.argmax(y_pred, dimension=1)



# ###########
# cost function
# calculating cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true)

# taking an average of cross entropy to get the cost
cost = tf.reduce_mean(cross_entropy)

# optimization step to reduce cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)


# ##########
# performance measures
# get how many predictions were correct and store in a boolean array
correct_predictions = tf.equal(y_pred_cls, y_true_cls)

# calculate accuracy by converting correct_pred to flaot and taking average
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))



# Graph ends
# ######################################################



# #########################
# utility functions
# #########################

def plot_images(images, cls_true, cls_pred=None):
	""" 
		PLotting 9 images conatined in images array
		The true and predicted and true classes are also plotted below the images
	"""
	# Dividing the plot into 9 subplots
	fig, axes = plt.subplots(3, 3)
	# adjusting spacing between subplots
	fig.subplots_adjust(wspace=0.3, hspace=0.3)

	# flatten each axes(subplot) and iterate over it
	for i, ax in enumerate(axes.flat):
		# plotting the images using subplot references
		ax.imshow(images[i].reshape(img_shape), cmap='binary')

		# selecting label for the image
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		# plotting the selected xlabel
		ax.set_xlabel(xlabel)

		# removing x and y ticks
		ax.set_xticks([])
		ax.set_yticks([])
	# show the plot
	# plt.show()


# helper functn to perform optimization
def optimize(num_of_iterations, batch_size):
	"""function to run optimizer for a fixed nmber of iterations"""

	for i in range(num_of_iterations):
		# get training data with specified batch size and train on it
		x_batch , y_true_batch = data.train.next_batch(batch_size)

		# creating the feed dict for training
		feed_dict_train = {x: x_batch, y_true: y_true_batch}

		# running our optimizer
		sess.run(optimizer, feed_dict=feed_dict_train)


# helper to show performance
def print_accuracy():
	"""displays accuracy of the model"""

	# create feed dict for accuracy
	feed_dict_test = {x: data.test.images, y_true_cls: data.test.cls}

	# run accuracy
	acc = sess.run(accuracy, feed_dict=feed_dict_test)
	# prinitng accuracy
	print("Accuracy on test set: {0}".format(acc))


# print error images
def print_example_errors():
	"""Display test images that were wrongly classified"""
	# creating feed dict
	feed_dict_test = {x: data.test.images, y_true_cls: data.test.cls}

	# getting a boolean array which signifies how many predictions are correct
	correct, cls_pred = sess.run([correct_predictions, y_pred_cls], feed_dict=feed_dict_test)

	# negating the correct bool array
	incorrect = (correct == False)

	# getting all the images that were classified wrongly
	wrong_images = data.test.images[incorrect]

	# getting wrongly predicted classes
	wrong_cls = cls_pred[incorrect]

	# getting their tru classes
	correct_cls = data.test.cls[incorrect]

	# plotting the first 9 images
	plot_images(wrong_images[0:9], correct_cls[0:9], wrong_cls[0:9])


# print error images
def print_correct_classifications():
	"""Display test images that were corrctly classified"""
	# creating feed dict
	feed_dict_test = {x: data.test.images, y_true_cls: data.test.cls}

	# getting a boolean array which signifies how many predictions are correct
	correct, cls_pred = sess.run([correct_predictions, y_pred_cls], feed_dict=feed_dict_test)

	# getting all the images that were classified correct
	correct_images = data.test.images[correct]

	# getting predicted classes
	pred_cls = cls_pred[correct]

	# getting true classes
	true_cls = data.test.cls[correct]

	# plotting the first 9 images
	plot_images(correct_images[0:9], pred_cls[0:9], true_cls[0:9])


def plot_weights():
	"""Print the weights for each class"""

	# get the weights
	W = sess.run(weights)

	# calulating min and max of weights to adjust the colors
	w_min = np.min(W)
	w_max = np.max(W)

	# dividing the plot into 3*4 subplots
	fig, axes = plt.subplots(3, 4)
	# adjusting spacing
	fig.subplots_adjust(wspace=0.3, hspace=0.3)

	for i, ax in enumerate(axes.flat):
		# only use weights for first 10 subplots (10 classes)
		if i < 10:
			# get the weight array and reshape it to size of image
			# the comma (,) operator helps in selecting coloumns
			image = W[:, i].reshape(img_shape)

			# set xlabels
			ax.set_xlabel("Weights: {0}".format(i))

			# plot the image
			# vmin and vmax help in normalizing the color over all the subplots
			ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

		# removing ticks
		ax.set_xticks([])
		ax.set_yticks([])

	# show the graph
	plt.show()






# ##############################################
# Running the Session
# ##############################################

with tf.Session() as sess:
	"""Inside session"""
	# Initialize all tensorflow variables
	sess.run(tf.global_variables_initializer())

	num_of_iterations = 1000
	batch_size = 5000

	# Run optimizer for said iterations
	optimize(num_of_iterations=num_of_iterations, batch_size=batch_size)

	# printing accuracy of model after training
	print_accuracy()

	print_example_errors()

	print_correct_classifications()

	plot_weights()

	plt.show()