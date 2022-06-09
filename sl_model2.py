import tensorflow as tf

NUM_INPUTS = 3
NUM_OUTPUTS = 1
HIDDEN1_UNITS = 50
HIDDEN2_UNITS = 50
HIDDEN3_UNITS = 50


# set up weight variable
def weight_variable(shape):
    initializer = tf.contrib.layers.xavier_initializer()
    initial = initializer(shape=shape)
    return tf.Variable(initial)


# set up bias variable
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# define hidden layer output
def hidden_layer(inputs, weights, name):
    return tf.nn.relu(tf.matmul(inputs, weights), name=name)


class SupervisedModel(object):
    """implements the supervised learning model"""
    def __init__(self):
        # placeholder inputs and labels
        self.x = tf.placeholder(tf.float32, shape=[None, NUM_INPUTS], name='x')
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1], name='labels')

        # fully connected layer 1
        self.W_fc1 = weight_variable([NUM_INPUTS, HIDDEN1_UNITS])
        self.h_fc1 = hidden_layer(self.x, self.W_fc1, 'fc1')

        # fully connected layer 2
        self.W_fc2 = weight_variable([HIDDEN1_UNITS, HIDDEN2_UNITS])
        self.h_fc2 = hidden_layer(self.h_fc1, self.W_fc2, 'fc2')

        # fully connected layer 2
        self.W_fc3 = weight_variable([HIDDEN2_UNITS, HIDDEN3_UNITS])
        self.h_fc3 = hidden_layer(self.h_fc2, self.W_fc3, 'fc3')

        # output layer
        self.W_fc4 = weight_variable([HIDDEN3_UNITS, NUM_OUTPUTS])
        self.b_fc4 = bias_variable([NUM_OUTPUTS])
        self.y = tf.tanh(tf.matmul(self.h_fc3, self.W_fc4) + self.b_fc4, name='y')










