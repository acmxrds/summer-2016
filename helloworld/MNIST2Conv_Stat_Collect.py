# coding=utf-8
__author__ = 'Abhineet Saxena'

"""

The Code for the ACM XRDS Hello World! column collects summary statistics for thethe CNN architecture constructed
from the architecture detailed at Google TensorFlow MNIST Expert tutorial:
https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html

Note:
The summary collection ops for most of the layers (Conv. Layer 1, Conv. Layer 2 and Softmax Layer) have
been commented out owing to a significant computation load that is entailed by the CPU for handling the summary
collection for all the layers at once. It can cripplingly slow down the machine while the file is in execution.
If you have a much better computing architecture than the one I use, you can certainly try running all ops at once:

My Configuration:
~~~~~~~~~~~~~~~~~~~~~~~~~~~
> Model Name: Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz
> No. of Processors: 3
> No. of CPU cores: 2
> Cache Size: 3072 KB
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instructions For Running TensorFlow:
For running the Tensorboard program and visualizing the statistics, anyone of the following commands needs to
be entered at the terminal and run:
>> tensorboard  --logdir='/path/to/mnist_logs folder'
or
>> python tensorflow/tensorboard/tensorboard.py --logdir='path/to/mnist_logs folder'

(Replace the string after the ‘=’ sign above with the actual path to the folder, without the single quotes.)

Thereafter, the TensorBoard panel can then be accessed by visiting the following URL in any of your browsers.
http://0.0.0.0:6006/

"""

# The Imports
import tensorflow as tf
# We make use of the script provided by the TensorFlow team for reading-in and processing the data.
import input_data as inpt_d


# ##Function Declarations
def weight_variable(shape, arg_name=None):
    """
    A method that returns a tf.Variable initialised with values drawn from a normal distribution.
    :param shape: The shape of the desired output.
    :return: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=arg_name)


def bias_variable(shape, arg_name=None):
    """
    A method that creates a constant Tensor with the specified shape and a constant value of 0.1.
    The bias value must be slightly positive to prevent neurons from becoming unresponsive or dead.
    :param shape: The shape of the desired output.
    :return: tf.Variable
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=arg_name)


def conv2d(xvar, Wt_var, stride_arg=(1, 1, 1, 1), pad_arg='SAME'):
    """
    Returns the Activation Map obtained by convolving the Weight matrix with the input matrix.
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

# Setting the Graph as the default Graph.
with new_graph.as_default():

    # Instantiating an Interactive Session.
    sess = tf.InteractiveSession()

    # Placeholder for the Input image data.
    xvar = tf.placeholder("float", shape=[None, 784], name="Input_Image")

    # Placeholder for the Input image label.
    y_var = tf.placeholder("float", shape=[None, 10], name="Input_Image_Label")

    # Setting up the variable that receives the processed MNIST dataset.
    mnist_data = inpt_d.read_data_sets('MNIST_data', one_hot=True)

    # ######The First Convolutional Layer #######

    # The Weight Matrix for the First Conv. Layer [28x28x32]. R=5, S=1, K=32 and P=2, The Input Channels: 1.
    # It has been named for use in collecting stats.
    Wt_mat_layer1 = weight_variable([5, 5, 1, 32], arg_name="Weights_Conv_Layer_1")

    # The Bias vector for the first Conv. Layer instantiated.
    bias_vec_layer1 = bias_variable([32], arg_name="Bias_Conv_Layer_1")

    # Reshapes the Image_Input into it's 28x28 matrix form. -1 implies flattening the image along the first dimension.
    x_image = tf.reshape(xvar, [-1, 28, 28, 1])

    # Convolution operation performed with scope as Conv_Layer_1 to aid visualization.
    with tf.name_scope("Conv_Layer_1") as scope_cv1:
        output_conv1 = tf.nn.relu(conv2d(x_image, Wt_mat_layer1) + bias_vec_layer1)
        pool_out_conv1 = max_pool_2x2(output_conv1)

    # Setting up the summary ops to collect the Weights, Bias and pool activation outputs.
    # Uncomment the following 3 lines for logging the outputs to summary op.
    # Wt_Cv1_summ = tf.histogram_summary("Conv1_Weights", Wt_mat_layer1)
    # Bs_Cv1_summ = tf.histogram_summary("Conv1_Bias", bias_vec_layer1)
    # Amap_Cv1_summ = tf.histogram_summary("Acivation_Max-Pooled_Output_Conv1", pool_out_conv1)

    # ######The Second Convolutional Layer #######

    # Instantiates the Weight Matrix defined per neuron for the second Conv. Layer [14x14x64]. R=5, K=64, S=1, P=2.
    # The Input channels: 32.
    Wt_mat_layer2 = weight_variable([5, 5, 32, 64], arg_name="Weights_Conv_Layer_2")
    bias_vec_layer2 = bias_variable([64], arg_name="Bias_Conv_Layer_2")

    # Operation of the second Conv. layer. Input has been padded (default).
    with tf.name_scope("Conv_Layer_2") as scope_cv2:
        output_conv2 = tf.nn.relu(conv2d(pool_out_conv1, Wt_mat_layer2) + bias_vec_layer2)
        pool_out_conv2 = max_pool_2x2(output_conv2)

    # Setting up the summary ops to collect the Weights, Bias and pool activation outputs.
    # Uncomment the following 3 lines for logging the outputs to summary op.
    # Wt_Cv2_summ = tf.histogram_summary("Conv2_Weights", Wt_mat_layer2)
    # Bs_Cv2_summ = tf.histogram_summary("Conv2_Bias", bias_vec_layer2)
    # Amap_Cv2_summ = tf.histogram_summary("Acivation_Max-Pooled_Output_Conv2", pool_out_conv2)

    # ######The First Fully Connected Layer #######

    # Weights initialised for the first fully connected layer. The FC layer has 1024 neurons.
    # The Output Volume from the previous layer has the structure 7x7x64.
    Wt_fc_layer1 = weight_variable([7 * 7 * 64, 1024], arg_name="Weights_FC_Layer")
    # Bias vector for the fully connected layer.
    bias_fc1 = bias_variable([1024], arg_name="Bias_FC_Layer")
    # The output matrix from 2nd Conv. layer reshaped to make it conducive to matrix multiply.
    # -1 implies flattening the Tensor matrix along the first dimension.
    pool_out_conv2_flat = tf.reshape(pool_out_conv2, [-1, 7*7*64])
    with tf.name_scope("FC_Layer") as scope_fc:
        output_fc1 = tf.nn.relu(tf.matmul(pool_out_conv2_flat, Wt_fc_layer1) + bias_fc1)

    # Setting up the summary ops to collect the Weights, Bias and pool activation outputs.
    Wt_FC_summ = tf.histogram_summary("FC_Weights", Wt_fc_layer1)
    Bs_FC_summ = tf.histogram_summary("FC_Bias", bias_fc1)
    Amap_FC_summ = tf.histogram_summary("Acivations_FC", output_fc1)

    # ##### Dropout #######
    # Placeholder for the Dropout probability.
    keep_prob = tf.placeholder("float", name="Dropout_Probability")
    # Performs the dropout op, where certain neurons are randomly disconnected and their outputs not considered.
    with tf.name_scope("CNN_Dropout_Op") as scope_dropout:
        h_fc1_drop = tf.nn.dropout(output_fc1, keep_prob)

    # ##### SoftMax-Regression #######
    W_fc2 = weight_variable([1024, 10], arg_name="Softmax_Reg_Weights")
    b_fc2 = bias_variable([10], arg_name="Softmax_Reg_Bias")
    # Performs the Softmax Regression op, computes the softmax probabilities assigned to each class.
    with tf.name_scope("Softmax_Regression") as scope_softmax:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # Setting up the summary ops to collect the Weights, Bias and pool activation outputs.
    # Uncomment the following 3 lines for logging the outputs to summary op.
    # Wt_softmax_summ = tf.histogram_summary("Sfmax_Weights", Wt_mat_layer2)
    # Bs_softmax_summ = tf.histogram_summary("Sfmax_Bias", bias_vec_layer2)
    # Amap_softmax_summ = tf.histogram_summary("Acivations_Sfmax", y_conv)

    # Cross-Entropy calculated.
    with tf.name_scope("X_Entropy") as scope_xentrop:
        cross_entropy = -tf.reduce_sum(y_var*tf.log(y_conv))
        # Adding the scalar summary operation for capturing the cross-entropy.
        ce_summ = tf.scalar_summary("Cross_Entropy", cross_entropy)

    # Adam Optimizer gives the best performance among Gradient Descent Optimizers.
    with tf.name_scope("Train") as scope_train:
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Calculating the Correct Prediction value.
    with tf.name_scope("Test") as scope_test:
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_var, 1))
        # The Bool tensor is converted or type-casted into float representation (1.s and 0s) and the mean for all the
        # values is calculated.
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # Adding the scalar summary operation for capturing the Accuracy.
        acc_summ = tf.scalar_summary("Accuracy", accuracy)

    # Adds the ops to the Graph that perform Variable initializations.
    # Merge all the summaries and write them out to /tmp/mnist_logs
    merged = tf.merge_all_summaries()
    summ_writer = tf.train.SummaryWriter("./mnist_logs", sess.graph_def)
    sess.run(tf.initialize_all_variables())

    # Training for 2000 iterations or Epochs.
    for i in range(2000):
        if i % 100 == 0:
            # Feeds the feed_dict dictionary with values from the test set.
            feed = {xvar: mnist_data.test.images, y_var: mnist_data.test.labels, keep_prob: 1.0}
            # The run method executes both the ops, i.e. 'merged' for merging the summaries and writing them
            # and the 'accuracy' op. for calculating the accuracy over the test set. Both are executed every
            # 100th iteration.
            result = sess.run([merged, accuracy], feed_dict=feed)
            # Summary string output obtained after the execution of 'merged' op.
            summary_str = result[0]
            # Accuracy value output obtained after the execution of 'accuracy' op.
            acc = result[1]
            # Adding the summary string and writing the output to the log-directory.
            summ_writer.add_summary(summary_str, i)
            print("Accuracy at step %s: %s" % (i, acc))
        else:
            # Returns the next 50 images and their labels from the training set.
            batch = mnist_data.train.next_batch(50)
            # Train the CNN with the dropout probability of neurons being 0.5 for every iteration.
            train_step.run(feed_dict={xvar: batch[0], y_var: batch[1], keep_prob: 0.5})
