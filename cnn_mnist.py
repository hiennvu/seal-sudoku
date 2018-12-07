from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import packages
import numpy as np 
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


# MODEL FUNCTION FOR CNN
# features is input feature maps
def cnn_model_fn(features, labels, mode):

	# Input layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) 
	# input has shape: [batch_size, image_height, image_width, channels]
	# -1 means 'flatten', will be dynamic with number of examples input (len(features))
	# 28x28 pixel images
	# e.g. is 5 examples are input, len(features["x"]) = 28*28*5 = 3920, which is reshaped to [5, 28, 28, 1]

	# Convolutional Layer 1
	# Each tf.layer method takes a tensor as input and outputs a transformed tensor
	# Applies 32 5x5 filters (extracting 5x5 pixel subregions) with ReLU activation function
	# conv1 shape = [batch_size, 28, 28, 32], increases to 32 channels, each an output from a filter
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 32,
		kernel_size = [5, 5],
		padding = "same",		
		# default = valid (truncates instead of padding)
		# pads the ends with 0s so output tensor is same size as input tensor
		# without padding, 5x5 convolution on a 28x28 tensor will produce a 24x24 tensor
		activation = tf.nn.relu)


	# Pooling Layer 1
	# Performs max pooling with 2x2 filter and stride of 2
	# pool1 shape = [batch_size, 14, 14, 32], filter reduces height and width by 50% each
	pool1 = tf.layers.max_pooling2d(
		inputs = conv1, 
		pool_size = [2, 2],
		strides = 2)

	# Convolutional layer 2
	# 64 5x5 filters, ReLU
	# conv2 shape = [batch_size, 14, 14, 64]	##Q: why 64 and not 64*32?
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 64,
		kernel_size = [5, 5],
		padding = "same",
		activation = tf.nn.relu)

	# Pooling layer 2
	# 2x2 filter, stride of 2
	# pool2 shape = [batch_size, 7, 7, 64]
	pool2 = tf.layers.max_pooling2d(
		inputs = conv2, 
		pool_size = [2, 2],
		strides = 2)

	# Dense layer 1
	# 1,024 neurons, with dropout regularization rate of 0.4 
	# (probability of 0.4 that any given element will be dropped during training - avoids overfitting)
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])	
	# flatten completely (pool2 shape = [batch_size, 7, 7, 64])
	dense = tf.layers.dense(
		inputs = pool2_flat,
		units = 1024,
		activation = tf.nn.relu)
	# dropout shape = [batch_size, 1024]
	dropout = tf.layers.dropout(
		inputs = dense,
		rate = 0.4,
		training = mode == tf.estimator.ModeKeys.TRAIN)
	

	# Dense layer 2
	# Logits layer. 10 neurons, one for each digit target class (0-9)
	# logits shape = [batch_size, 10]
	logits = tf.layers.dense(
		inputs = dropout,
		units = 10)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input = logits, axis=1),
		# Add 'softmax_tensor' (derive probabilities) to the graph. Used for PREDICT and by 'logging_hook' (?)
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}


	# Return prediction (for PREDICT mode)
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes) EVAL = TEST
	# softmax crossentropy/categorial crossentropy comparing labels with predictions
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	# optimising loss during training
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# LOAD TRAINING AND TEST DATA
def main(unused_argv):
	mnist = tf.contrib.learn.datasets.load_dataset("mnist")
	# Load training data
	train_data = mnist.train.images # Returns np.array with 0s and 1s
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	# Load test data
	eval_data = mnist.test.images # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

	# Create the Estimator (tf class for performing high-level model training, evaluation, and inference)
	mnist_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, 	# model function above
		model_dir="/tmp/mnist_convnet_model") # where model data will be saved

	# Set up logging for predictions (since CNNs can take a while to train)
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=50)		# log every 50 steps in training

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": train_data},
		y=train_labels,
		batch_size=100,		# trains on mini batches of 100 at each step
		num_epochs=None,	# model will train until the specified number of steps is reached
		shuffle=True)		# shuffle training data
	mnist_classifier.train(
		input_fn=train_input_fn,
		steps=1,		# train for 2000 steps total (tutorial said 20k steps but my laptop is struggling to handle it)
		hooks=[logging_hook])	# logging

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": eval_data},
		y=eval_labels,
		num_epochs=1,		# Q: what is epoch and how does this differ to batchsize?
		shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)


if __name__ == "__main__":
  tf.app.run()