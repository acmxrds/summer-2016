__author__ = 'Abhineet Saxena'

"""
The following Code for the ACM XRDS Hello World! column details the default architecture of the CNN constructed
in Google TensorFlow MNIST Expert tutorial:
https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html
The Code has been refined with the addition of explanations as comments.
The function structures have also been improved upon.
"""
# The Imports
import tensorflow as tf
import numpy as np
# We make use of the script provided by the TensorFlow team for reading-in and processing the data.
import input_data as inpt_d


# ##Function Declarations
def weight_variable(shape):
    """
    A method that returns a tf.Variable initialised with values drawn from a normal distribution.
    :param shape: The shape of the desired output.
    :return: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    A method that creates a constant Tensor with the specified shape and a constant value of 0.1.
    The bias value must be slightly positive to prevent neurons from becoming unresponsive or dead.
    :param shape: The shape of the desired output.
    :return: tf.Variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(xvar, Wt_var, stride_arg=(1, 1, 1, 1), pad_arg='SAME'):
    """
    returns the Activation Map obtained by convolving the Weight matrix with the input matrix.
    :param xvar: The Neural Input Matrix.
    :param Wt_var: The Weight Matrix.
    :param stride_arg: The Stride value, specified as a tuple.
    :param pad_arg: The Padding Value. Can either be 'VALID' (padding disabled) or 'SAME' (padding-enabled).
    :return: The Activation Map or the Output Volume.
    """
    return tf.nn.conv2d(xvar, Wt_var, strides=[sval for sval in stride_arg], padding=pad_arg)


def max_pool_2x2(xvar):
    """
    Performs the max-pooling operation. Here, a default window size of 2x2 and stride values of (2, 2) is assumed.
    :param xvar: The Input Volume to be max-pooled.
    :return: Teh max-pooled output.
    """
    return tf.nn.max_pool(xvar, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Creating a Graph
new_graph = tf.Graph()

# Setting the Graph as the default Graph. The as_default() method returns a context manager, i.e. brings the focus to
# current graph. This is useful if we have multiple graphs. However, I make use of it here to demonstrate it's usage.
with new_graph.as_default():
    # Two types of Session can be instantiated: a regular one and the Interactive one.
    # The latter one installs itself as the default session. Thus for ease of use, we use the latter.
    sess = tf.InteractiveSession()

    # The format of data, that is being read for a single image, is a 1-D vector of size 28x28 = 784 units.
    # A placeholder is simply a variable definition for data that isn't specified immediately but will be done at a
    # later stage.
    xvar = tf.placeholder("float", shape=[None, 784], name="Input_Image")

    # It signifies the format for the label data converted to a 1-hot encoding. For digits 0 to 9, 10 possible
    # classes exist, hence the shape. Thus for each image, we have a vector of size 10 consisting of zeroes for digits
    # which the image does not represent and 1 for the digit that it does.
    y_var = tf.placeholder("float", shape=[None, 10], name="Input_Image_Label")

    # Setting up the variable that receives the processed MNIST dataset.
    mnist_data = inpt_d.read_data_sets('MNIST_data', one_hot=True)

    # ######The First Convolutional Layer #######

    # Instantiates the Weight Matrix defined per neuron for the first Conv. Layer. We specify the Receptive Field
    # here [5x5]and the value of K, or the depth of the Convolutional layer: 32. 1 specifies the no. of colour channels.
    Wt_mat_layer1 = weight_variable([5, 5, 1, 32])

    # The Bias vector for the first Conv. Layer instantiated. The size of the vector is equal to that
    # of the depth value, K.
    bias_vec_layer1 = bias_variable([32])

    # Reshapes the Image_Input into it's 28x28 matrix form. -1 implies flattening the image along the first dimension.
    x_image = tf.reshape(xvar, [-1, 28, 28, 1])

    # Here, we perform the convolutional op b/w the image and the Kernel Weight Matrix. Denotes the operation of the
    # first Conv. layer. The input has been padded (default).
    # The neural outputs are all passed through ReLu activation function and max-pooled.
    output_conv1 = tf.nn.relu(conv2d(x_image, Wt_mat_layer1) + bias_vec_layer1)
    pool_out_conv1 = max_pool_2x2(output_conv1)

    # ######The Second Convolutional Layer #######

    # Instantiates the Weight Matrix defined per neuron for the second Conv. Layer. Receptive Field value: [5x5],
    # The Input channels: 32, Output channels or Depth (K) = 64.
    Wt_mat_layer2 = weight_variable([5, 5, 32, 64])
    bias_vec_layer2 = bias_variable([64])

    # Operation of the second Conv. layer. Input has been padded (default).
    output_conv2 = tf.nn.relu(conv2d(pool_out_conv1, Wt_mat_layer2) + bias_vec_layer2)
    pool_out_conv2 = max_pool_2x2(output_conv2)

    # ######The First Fully Connected Layer #######

    # Weights initialised for the first fully connected layer. The FC layer has 1024 neurons.
    # The Output Volume from the previous layer has the structure 7x7x64.
    Wt_fc_layer1 = weight_variable([7 * 7 * 64, 1024])
    # Bias vector for the fully connected layer.
    bias_fc1 = bias_variable([1024])
    # The output matrix from 2nd Conv. layer reshaped to make it conducive to matrix multiply.
    # -1 implies flattening the Tensor matrix along the first dimension.
    pool_out_conv2_flat = tf.reshape(pool_out_conv2, [-1, 7*7*64])
    output_fc1 = tf.nn.relu(tf.matmul(pool_out_conv2_flat, Wt_fc_layer1) + bias_fc1)

    # ##### Dropout #######
    # Placeholder for the probability that a neuron output would be retained during dropout operation.
    keep_prob = tf.placeholder("float")
    # Performs the dropout op, where certain neurons are randomly disconnected and their outputs not considered.
    # It's effective at controlling over-fitting of the CNN to training data.
    h_fc1_drop = tf.nn.dropout(output_fc1, keep_prob)

    # ##### SoftMax-Regression #######
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    # Performs the Softmax Regression op, computes the softmax probabilities assigned to each class.
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Cross-Entropy calculated.
    cross_entropy = -tf.reduce_sum(y_var*tf.log(y_conv))

    # Adam Optimizer gives the best performance among Gradient Descent Optimizers.
    # train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

    # tf.equal() returns the truth values for the equal-to operation performed in an element-wise fashion for the
    # two tensors being compared. tf.argmax() returns the index with the largest value across the dimension
    # specified as the second argument, i.e. the first dimension in this case.
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_var, 1))

    # The Bool tensor is converted or type-casted into float representation (1.s and 0s) and the mean for all the
    # values is calculated.
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Adds the ops to the Graph that perform Variable initializations.
    sess.run(tf.initialize_all_variables())

    # Training for 2000 iterations or Epochs.
    for i in range(2000):
        # Returns the next 50 images and their labels from the train set.
        batch = mnist_data.train.next_batch(50)
        if i % 100 == 0:
            # For every 100th training iteration, invoke the accuracy.eval() method that takes as input a feed_dict
            # that maps the placeholders we declared earlier into actual values of the batch.
            train_accuracy = accuracy.eval(feed_dict={xvar: batch[0], y_var: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
        # Train the CNN with the dropout probability of neurons being 0.5 for every iteration.
        train_step.run(feed_dict={xvar: batch[0], y_var: batch[1], keep_prob: 0.5})
    # Prints the accuracy of the CNN by running over the test set.
    print "test accuracy %g" % accuracy.eval(feed_dict={xvar: mnist_data.test.images,\
                                                        y_var: mnist_data.test.labels, keep_prob: 1.0})