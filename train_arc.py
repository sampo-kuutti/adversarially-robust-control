import numpy as np
import os
import tensorflow as tf
import ctypes
import csv
import random
import ipg_proxy
from collections import deque
import time
import datetime
import argparse
import sl_network
import threading

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
# PARAMETERS
FPATH = '/vol/research/safeav/Sampo/condor-a2c/test/log_arc/'  # use project directory path
LOG_DIR = timestamp  # save location for logs
MODEL_PATH = './models/'
ARL_MODEL = 'c111386_p2/model-ep-2500-finalr-3339.ckpt'
ARL_MODELS = ['c48602_p1/model-ep-2500-finalr-3760.ckpt', 'c117923_p0/model-ep-2500-finalr-3771.ckpt',
              'c117926_p0/model-ep-2500-finalr-761.ckpt', 'c111386_p2/model-ep-2500-finalr-3339.ckpt',
              'c48603_p10/model-ep-2500-finalr-3785.ckpt']
IL_MODEL = 'sl_models/model-step-901000-val-0.0150463.ckpt'
N_WORKERS = 5  # number of workers
MAX_EP_STEP = 200  # maximum number of steps per episode (unless another limit is used)
MAX_GLOBAL_EP = 1000  # total number of episodes
MAX_PROXY_EP = 1000  # total number of episodes to train on proxy, before switching to ipg simulations
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 64  # sets how often the global net is updated
GAMMA = 0.99  # discount factor
ENTROPY_BETA = 1e-4  # entropy factor
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-2  # learning rate for critic
LR_P = 1e-5  # learning rate for protagonist
SAFETY_ON = 0  # safety cages, 0 = disabled 1 = enabled
REPLAY_MEMORY_CAPACITY = int(1e4)  # capacity of experience replay memory
TRAUMA_MEMORY_CAPACITY = int(1e2)  # capacity of trauma memory
MINIBATCH_SIZE = 64  # size of the minibatch for training with experience replay
TRAJECTORY_LENGTH = 80  # size of the trajectory used in weight updates
UPDATE_ENDSTEP = True  # update at the end of episode using previous MB_SIZE experiences
UPDATE_TRAUMA = 16  # update weights using the trauma memory every UPDATE_TRAUMA updates
OFF_POLICY = True  # update off-policy using ER/TM
ON_POLICY = True  # update on-policy using online experiences
CHECKPOINT_EVERY = 100  # sets how often to save weights
HN_A = 50  # hidden neurons for actor network
HN_C = 50  # hidden neurons for critic network
LSTM_UNITS = 16  # lstm units in actor network
MAX_GRAD_NORM = 0.5  # max l2 grad norm for gradient clipping
V_MIN = 12  # minimum lead vehicle velocity (m/s)
V_MAX = 30  # maximum lead vehicle velocity (m/s)
LAMBDA_P = 100
C_TH = 0  # if non-zero, only applied protagonist distillation loss at t_h > C_TH
C_AP = 0.5
# Action Space Shape
N_S = 4  # number of states
N_S_P = 3
N_A = 1  # number of actions
A_BOUND = [-6, 2]  # action bounds


def get_arguments():
    parser = argparse.ArgumentParser(description='RL training')
    parser.add_argument(
        '--num_advs',
        type=int,
        default=N_WORKERS,
        help='Number of adversaries for ARC training.'
    )
    parser.add_argument(
        '--lr_a',
        type=float,
        default=LR_A,
        help='Actor learning rate'
    )
    parser.add_argument(
        '--lr_c',
        type=float,
        default=LR_C,
        help='Critic learning rate'
    )
    parser.add_argument(
        '--lr_p',
        type=float,
        default=LR_P,
        help='Protagonist learning rate'
    )
    parser.add_argument(
        '--lambda_p',
        type=float,
        default=LAMBDA_P,
        help='Protagonist converativism cost'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=GAMMA,
        help='Discount rate gamma'
    )
    parser.add_argument(
        '--max_eps',
        type=int,
        default=MAX_GLOBAL_EP,
        help='Checkpoint file to restore model weights from.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=MINIBATCH_SIZE,
        help='Batch size. Must divide evenly into dataset sizes.'
    )
    parser.add_argument(
        '--trajectory',
        type=float,
        default=TRAJECTORY_LENGTH,
        help='Length of trajectories in minibatches'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=CHECKPOINT_EVERY,
        help='Number of steps before checkpoint.'
    )
    parser.add_argument(
        '--ent_beta',
        type=float,
        default=ENTROPY_BETA,
        help='Entropy coefficient beta'
    )
    parser.add_argument(
        '--fpath',
        type=str,
        default=FPATH,
        help='File path to root folder.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=LOG_DIR,
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--store_metadata',
        type=bool,
        default=False,
        help='Storing debug information for TensorBoard.'
    )
    parser.add_argument(
        '--restore_from',
        type=str,
        default=None,
        help='Checkpoint file to restore model weights from.'
    )
    parser.add_argument(
        '--hn_a',
        type=int,
        default=HN_A,
        help='Number of hidden neurons in actor network.'
    )
    parser.add_argument(
        '--hn_c',
        type=int,
        default=HN_C,
        help='Number of hidden neurons in critic network.'
    )
    parser.add_argument(
        '--lstm_units',
        type=int,
        default=LSTM_UNITS,
        help='Number of lstm cells in actor network.'
    )
    parser.add_argument(
        '--store_results',
        action='store_true',
        help='Storing episode results in csv files.'
    )
    parser.add_argument(
        '--trauma',
        action='store_true',
        help='If true use trauma memory in off-policy updates.'
    )
    parser.add_argument(
        '--max_norm',
        type=float,
        default=MAX_GRAD_NORM,
        help='Maximum L2 norm of the gradient for gradient clipping.'
    )
    parser.add_argument(
        '--c_th',
        type=float,
        default=C_TH,
        help='Time Headway Constant. If non-zero, only applies protagonist distillation loss at t_h > C_TH'
    )
    parser.add_argument(
        '--c_ap',
        type=float,
        default=C_AP,
        help='Action Constant for Conservative Driving Loss'
    )
    parser.add_argument(
        '--v_min',
        type=int,
        default=V_MIN,
        help='Minimum lead vehicle velocity (m/s).'
    )
    parser.add_argument(
        '--v_max',
        type=int,
        default=V_MAX,
        help='Maximum lead vehicle velocity (m/s).'
    )

    return parser.parse_args()


def calculate_reward(th, delta_th, x_rel):
    if 0 <= th < 0.50:  # crash imminent
        reward = -10
    elif 0.50 <= th < 1.75 and delta_th <= 0:  # too close
        reward = -0.5
    elif 0.50 <= th < 1.75 and delta_th > 0:  # closing up
        reward = 0.1
    elif 1.75 <= th < 1.90:  # goal range large
        reward = 0.5
    elif 1.90 <= th < 2.10:  # goal range small
        reward = 5
    elif 2.10 <= th < 2.25:  # goal range large
        reward = 0.5
    elif 2.25 <= th < 10 and delta_th <= 0:  # closing up
        reward = 0.1
    elif 2.25 <= th < 10 and delta_th > 0:  # too far
        reward = -0.1
    elif th >= 10 and delta_th <= 0:  # closing up
        reward = 0.05
    elif th >= 10 and delta_th > 0:  # way too far
        reward = -10
    elif x_rel <= 0:
        reward = -100  # crash occurred
    else:
        print('no reward statement requirements met (th = %f, delta_th = %f, x_rel = %f), reward = 0'
              % (th, delta_th, x_rel))
        reward = 0

    return reward


def calculate_reward2(th, delta_th, x_rel):
    if 0 <= th < 0.50:  # crash imminent
        reward = 0.5
    elif 0.50 <= th < 1.75 and delta_th <= 0:  # too close
        reward = 0.1
    elif 0.50 <= th < 1.75 and delta_th > 0:  # closing up
        reward = -0.1
    elif 1.75 <= th < 1.90:  # goal range large
        reward = -0.5
    elif 1.90 <= th < 2.10:  # goal range small
        reward = -1
    elif 2.10 <= th < 2.25:  # goal range large
        reward = -0.5
    elif 2.25 <= th < 10 and delta_th <= 0:  # closing up
        reward = -0.5
    elif 2.25 <= th < 10 and delta_th > 0:  # too far
        reward = -0.5
    elif th >= 10 and delta_th <= 0:  # closing up
        reward = -0.5
    elif th >= 10 and delta_th > 0:  # way too far
        reward = -0.5
    elif x_rel <= 0:
        reward = 1  # crash occurred
    else:
        print('no reward statement requirements met (th = %f, delta_th = %f, x_rel = %f), reward = 0'
              % (th, delta_th, x_rel))
        reward = 0

    return reward


# reward function based on inverse time headway and large pay off for crashes
def calculate_reward3(t_h):
    if t_h > 0:  # positive time headway
        r = 1 / t_h
    else:  # crash occurred
        r = 100

    # cap r at 100 and 0.5 (for t_h < 0.01s and t_h > 2)
    if r > 100:
        r = 100
    elif r < 0.5:
        r = 0.5

    return r


class ModelAdv(object):
    def __init__(self, scope, sess, globalAC=None):
        self.args = get_arguments()
        self.sess = sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(self.args.lr_a, name='RMSPropA')  # optimizer for the actor
        self.critic_optimizer = tf.train.RMSPropOptimizer(self.args.lr_c, name='RMSPropC')  # optimizer for the critic
        self.prot_optimiser = tf.train.RMSPropOptimizer(self.args.lr_p, name='RMSPropP')

        if scope == GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.x = tf.placeholder(tf.float32, [None, N_S_P], 'prot_s')
                self.p_params = self._build_net(scope)[-1]
                self.restore_models(scope)

        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # state
                self.a_his = tf.placeholder(tf.float32, [None, 1], 'A')  # action
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')  # v_target value

                # prot net placeholders
                self.x = tf.placeholder(tf.float32, [None, N_S_P], 'prot_s')
                self.p_labels = tf.placeholder(tf.float32, [None, 1], 'prot_labels')
                self.y = tf.placeholder(tf.float32, [None, 1], 'prot_a')

                self.mu, self.sigma, self.v, self.a_p, self.a_params, self.c_params, self.p_params = self._build_net(
                    scope)
                self.restore_models(scope)

                # advantage function A(s) = V_target(s) - V(s)
                td = tf.subtract(self.v_target, self.v, name='TD_error')

                # Critic Loss
                with tf.name_scope('c_loss'):
                    # value loss L = (R - V(s))^2
                    self.c_loss = tf.reduce_mean(tf.square(td))

                # Scale mu to action space, and add small value to sigma to avoid NaN errors
                with tf.name_scope('wrap_a_out'):
                    # use abs value of A_BOUND[0] as it is bigger than A_BOUND[1]
                    # The action value is later clipped so values outside of A_BOUND[1] will be constrained
                    self.mu, self.sigma = self.mu * (-A_BOUND[0]), self.sigma + 1e-4

                # Normal distribution with location = mu, scale = sigma
                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

                # Actor loss
                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    # Entropy H(s) = 0.5(log(2*pi*sigma^2)+1) see: https://arxiv.org/pdf/1602.01783.pdf page 13
                    entropy = normal_dist.entropy()  # encourage exploration
                    # policy loss L = A(s,a) * -logpi(a|s) - B * H(s)
                    self.a_loss = tf.reduce_mean(-(self.args.ent_beta * entropy + log_prob * td))
                    self.p_penalty = loss = tf.reduce_mean(tf.losses.absolute_difference(
                        labels=self.p_labels, predictions=self.a_p))  # penalty for overly conservative driving
                    self.p_loss = tf.reduce_mean(
                        (self.args.ent_beta * entropy + log_prob * td) + (self.args.lambda_p * self.p_penalty))

                # Choose action
                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0],
                                              A_BOUND[1])  # sample a action from distribution

                # Compute the gradients
                with tf.name_scope('local_grad'):
                    # calculate gradients for the network weights
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    self.p_grads = tf.gradients(self.p_loss, self.p_params)
                    # clip gradients by global norm
                    self.a_grads, a_grad_norm = tf.clip_by_global_norm(self.a_grads, MAX_GRAD_NORM)
                    self.c_grads, c_grad_norm = tf.clip_by_global_norm(self.c_grads, MAX_GRAD_NORM)
                    self.p_grads, p_grad_norm = tf.clip_by_global_norm(self.p_grads, MAX_GRAD_NORM)

                # Update weights

                with tf.name_scope('sync'):  # update local and global network weights
                    with tf.name_scope('pull'):
                        self.pull_p_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.p_params, globalAC.p_params)]
                    with tf.name_scope('push'):
                        # actor and critic are updated locally
                        self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, self.a_params))
                        self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, self.c_params))
                        # prot is updated to global network
                        self.update_p_op = self.prot_optimiser.apply_gradients(zip(self.p_grads, self.p_params))

    # Build the network
    def _build_net(self, scope):  # neural network structure of the actor and critic

        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('prot'):
            l1 = tf.layers.dense(self.x, 50, tf.nn.relu, use_bias=False, name='l1')
            l2 = tf.layers.dense(l1, 50, tf.nn.relu, use_bias=False, name='l2')
            l3 = tf.layers.dense(l2, 50, tf.nn.relu, use_bias=False, name='l3')
            a = tf.layers.dense(l3, 1, tf.nn.tanh, name='a')

        prot_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/prot')
        # return a, prot_params

        obs = tf.concat([self.s, a], 1)
        # Actor network
        with tf.variable_scope('actor'):
            # hidden layer
            l1_a = tf.layers.dense(obs, self.args.hn_a, tf.nn.relu6, kernel_initializer=w_init, name='l1a')
            l2_a = tf.layers.dense(l1_a, self.args.hn_a, tf.nn.relu6, kernel_initializer=w_init, name='l2a')
            l3_a = tf.layers.dense(l2_a, self.args.hn_a, tf.nn.relu6, kernel_initializer=w_init, name='l3a')

            # Recurrent network for temporal dependencies
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.args.lstm_units, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(l3_a, [0])
            step_size = tf.shape(obs)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, self.args.lstm_units])

            # expected action value
            mu = tf.layers.dense(rnn_out, N_A, tf.nn.tanh, kernel_initializer=w_init,
                                 name='mu')  # estimated action value
            # expected variance
            sigma = tf.layers.dense(rnn_out, N_A, tf.nn.softplus, kernel_initializer=w_init,
                                    name='sigma')  # estimated variance

        # Critic network
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(obs, self.args.hn_c, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l2_c = tf.layers.dense(l_c, self.args.hn_c, tf.nn.relu6, kernel_initializer=w_init, name='l2_c')
            v = tf.layers.dense(l2_c, 1, kernel_initializer=w_init, name='v')  # estimated value for state

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a, a_params, c_params, prot_params

    def update_global(self, feed_dict):  # run by a local
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def update_crit(self, feed_dict):  # run by a local
        self.sess.run(self.update_c_op, feed_dict)  # local grads applies to global net

    def choose_action(self, s, x, rnn_state):  # run by a local
        # reshape state vectors
        s = np.reshape(s, (1, N_S))
        x = np.reshape(x, (1, N_S_P))
        # compute new LSTM cell state
        rnn_state = self.sess.run(self.state_out, {self.s: s, self.x: x,
                                                   self.state_in[0]: rnn_state[0],
                                                   self.state_in[1]: rnn_state[1]})
        # infer action
        a = self.sess.run(self.A, {self.s: s, self.x: x,
                                   self.state_in[0]: rnn_state[0],
                                   self.state_in[1]: rnn_state[1]
                                   })[0]
        return a, rnn_state

    def choose_action_prot(self, s):
        s = np.reshape(s, (1, N_S_P))
        return self.sess.run(self.a_p, {self.x: s})[0]

    def train_prot(self, feed_dict):
        self.sess.run(self.update_p_op, feed_dict)  # local grads applies to global net

    def pull_prot(self):  # pull global prot params to local network
        self.sess.run(self.pull_p_params_op)

    def restore_models(self, scope):
        # Choose arl model, loops over the model initialisations in ARL_MODELS
        if scope == GLOBAL_NET_SCOPE or scope == 'W_0':
            arl_model = ARL_MODELS[0]
        else:
            worker_num = int(scope[-1])
            n_models = len(ARL_MODELS)
            arl_model = ARL_MODELS[worker_num % n_models]
        # This fix works around the Type Error that comes when I tried to use tf.saver to initiate from checkpoint
        # which complained about missing variables in checkpoint since the graph adds ":0" to the end of variables
        # but when building an assignment map the ":0" can be ignored which results in the very stupid, but effective,
        # solution below....
        arl_varlist = {'W_0/actor/l1a/bias': scope + '/actor/l1a/bias',
                       'W_0/actor/l1a/kernel': scope + '/actor/l1a/kernel',
                       'W_0/actor/l2a/bias': scope + '/actor/l2a/bias',
                       'W_0/actor/l2a/kernel': scope + '/actor/l2a/kernel',
                       'W_0/actor/l3a/bias': scope + '/actor/l3a/bias',
                       'W_0/actor/l3a/kernel': scope + '/actor/l3a/kernel',
                       'W_0/actor/mu/bias': scope + '/actor/mu/bias',
                       'W_0/actor/mu/kernel': scope + '/actor/mu/kernel',
                       'W_0/actor/rnn/lstm_cell/bias': scope + '/actor/rnn/lstm_cell/bias',
                       'W_0/actor/rnn/lstm_cell/kernel': scope + '/actor/rnn/lstm_cell/kernel',
                       'W_0/actor/sigma/bias': scope + '/actor/sigma/bias',
                       'W_0/actor/sigma/kernel': scope + '/actor/sigma/kernel',
                       'W_0/critic/lc/bias': scope + '/critic/lc/bias',
                       'W_0/critic/lc/kernel': scope + '/critic/lc/kernel',
                       'W_0/critic/l2_c/bias': scope + '/critic/l2_c/bias',
                       'W_0/critic/l2_c/kernel': scope + '/critic/l2_c/kernel',
                       'W_0/critic/v/bias': scope + '/critic/v/bias',
                       'W_0/critic/v/kernel': scope + '/critic/v/kernel'}

        il_varlist = {'Variable': scope + '/prot/l1/kernel',
                      'Variable_1': scope + '/prot/l2/kernel',
                      'Variable_2': scope + '/prot/l3/kernel',
                      'Variable_3': scope + '/prot/a/kernel',
                      'Variable_4': scope + '/prot/a/bias'
                      }

        tf.train.init_from_checkpoint(str(MODEL_PATH + arl_model),
                                      arl_varlist)

        print('Initialised ARL model from: ', str(MODEL_PATH + arl_model))

        tf.train.init_from_checkpoint(str(MODEL_PATH + IL_MODEL),
                                      il_varlist)

        print('Initialised IL model from: ', str(MODEL_PATH + IL_MODEL))


# worker class that inits own environment, trains on it and updloads weights to global net
class Worker(object):
    def __init__(self, args, name, globalAC, sess, proxy, sl_net):

        self.name = name
        self.sess = sess
        self.adv_net = ModelAdv(name, self.sess, globalAC)
        self.args = args
        self.summary_writer = tf.summary.FileWriter(args.fpath + args.log_dir + '/' + str(self.name), sess.graph)
        self.proxy = proxy
        self.sl_net = sl_net

        # with sess.as_default():
        #    self.saver = tf.train.Saver()
        #    tf.global_variables_initializer().run()
        #    self.adv_net.restore_models()

    def work(self):
        global rewards, episodes, crashes
        total_step = 1

        buffer_s_p, buffer_s_a, buffer_a_p, buffer_a_a, buffer_r_a, buffer_p_l = [], [], [], [], [], []
        arr_scen = []  # driving scenarios

        # define tensorboard scalars and histograms
        if self.name == 'W_0':  # track losses and actions of worker #0
            sum_ploss = tf.summary.scalar('loss/policy_loss', self.adv_net.a_loss)
            sum_vloss = tf.summary.scalar('loss/value_loss', self.adv_net.c_loss)
            sum_mu = tf.summary.histogram('mu', self.adv_net.mu)
            sum_sigma = tf.summary.histogram('sigma', self.adv_net.sigma)
            sum_v = tf.summary.histogram('v', self.adv_net.v)
            sum_vt = tf.summary.histogram('v_target', self.adv_net.v_target)
            sum_a = tf.summary.histogram('act_out', self.adv_net.A)
            sum_ap = tf.summary.histogram('act_prot', self.adv_net.a_p)

        # loop episodes
        while not coord.should_stop() and episodes < args.max_eps:
            print('Episode: ', episodes)
            step = 0

            # initialise rnn state
            rnn_state = self.adv_net.state_init
            batch_rnn_state = rnn_state

            # set states to zero
            b = 0
            v_rel = 0
            v = 0
            x_rel = 0
            a = 0
            t = 0
            t_h = 0

            # empty arrays
            arr_a = []  # acceleration array
            arr_j = []  # jerk array
            arr_t = []  # time array
            arr_x = []  # x_rel array
            arr_v = []  # velocity array
            arr_dv = []  # relative velocity array
            arr_th = []  # time headway array
            arr_y_0 = []  # original output
            arr_y_sc = []  # safety cage output
            arr_sc = []  # safety cage number
            arr_cof = []  # coefficient of friction

            arr_v_leader = []  # lead vehicle velocity
            arr_a_leader = []  # lead vehicle acceleration

            arr_rewards = []  # rewards list

            # lead vehicle states
            T_lead = []
            X_lead = []
            V_lead = []
            A_lead = []

            # load test run
            # Option 1: Use random coefficients of frictions
            scen = random.randint(1, 25)
            arr_scen.append(scen)
            cof = 0.375 + scen * 0.025  # calculate coefficient of friction

            ep_r = 0

            # set initial states
            t = 0
            v = 25.5  # 91.8 km/h
            a = 0
            # x = random.randint(0, 5)
            x = 5

            # lead vehicle states
            x_lead = 55  # longitudinal position
            v_lead = 24 + random.randint(1, 8)  # velocity, randomly chosen between 25 and 32 m/s
            a_lead = 0  # acceleration
            v_rel = v_lead - v  # relative velocity
            x_rel = x_lead - x  # relative distance
            if v != 0:  # check for division by 0
                t_h = x_rel / v
            else:
                t_h = x_rel

            crash = 0  # variable for checking if a crash has occurred (0=no crash, 1=crash)
            too_far = 0  # is the lead vehicle too far?
            prev_output = 0

            while t < 300 and crash == 0 and too_far == 0:  # run sim for 300s or until crash occurs
                b += 1
                s_p = [v_rel, t_h, v]  # protagonist states
                s_a = [v_rel, t_h, v, a]  # adversary states

                action_prot = self.adv_net.choose_action_prot(s_p)
                action_adv, rnn_state = self.adv_net.choose_action(s_a, s_p, rnn_state)

                a_lead_ = float(action_adv)  # estimate stochastic action based on policy
                v_lead_ = v_lead + (a_lead * 0.04)
                x_lead_ = x_lead + (v_lead * 0.04)
                # constraints
                if v_lead_ > args.v_max:
                    v_lead_ = float(args.v_max)
                elif v_lead_ < args.v_min:
                    v_lead_ = float(args.v_min)

                arr_y_0.append(float(action_prot))

                output = action_prot
                sc = 0

                arr_y_sc.append(float(output))
                arr_sc.append(sc)

                # read new states
                # read host states
                t_ = t + 0.04  # time
                proxy_out = self.proxy.inference([v, a, cof, output, prev_output])  # proxy_out infers the v_t+1
                v_ = float(proxy_out)  # host velocity
                delta_v = v_ - v  # calculate delta_v
                if delta_v > 0.4:  # limit a to +/- 10m/s^2
                    delta_v = 0.4
                    v_ = delta_v + v
                elif delta_v < -0.4:
                    delta_v = -0.4
                    v_ = delta_v + v
                if v_ < 0:  # check for negative velocity
                    v_ = 0
                a_ = delta_v / 0.04  # host longitudinal acceleration
                x_ = x + (v * 0.04)  # host longitudinal position
                # print('t = %f, y = %f, v = %f, a = %f, x = %f' % (t, output, v, a, x))

                # relative states
                v_rel_ = float(v_lead_ - v_)  # relative velocity
                x_rel_ = float(x_lead_ - x_)  # relative distance

                # enter variables into arrays
                arr_a.append(a)
                arr_t.append(t)
                arr_x.append(x_rel)
                arr_v.append(v)
                arr_dv.append(v_rel)
                arr_th.append(t_h)
                arr_cof.append(cof)

                arr_v_leader.append(v_lead)
                arr_a_leader.append(a_lead)

                # calculate time headway
                if v_ != 0:
                    t_h_ = x_rel_ / v_
                else:
                    t_h_ = x_rel_

                # calculate reward
                reward = calculate_reward3(t_h_)
                ep_r += reward
                arr_rewards.append(reward)

                # stop simulation if a crash occurs
                if x_rel_ <= 0:
                    crash = 1
                    crashes += 1
                    print('crash occurred: simulation run stopped')
                elif t_h_ > 15:
                    too_far = 1
                    print('too far from lead vehice: simulation run stopped')

                # calculate "label" for protagonist to avoid overly conservative behaviour
                if t_h_ > args.c_th:
                    p_label = self.sl_net.inference(s_p)
                else:
                    p_label = action_prot

                buffer_s_p.append(s_p)
                buffer_s_a.append(s_a)
                buffer_a_p.append(action_prot)
                buffer_a_a.append(action_adv)
                buffer_r_a.append(reward)
                buffer_p_l.append(p_label)

                # update weights
                if total_step % UPDATE_GLOBAL_ITER == 0:  # update global and assign to local net
                    if t == 300 or crash == 1 or too_far == 1:
                        v_s_ = 0  # terminal state
                    else:
                        v_s_ = sess.run(self.adv_net.v, {self.adv_net.s: np.reshape(s_a, (1, N_S)),
                                                         self.adv_net.x: np.reshape(s_p, (1, N_S_P))})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r_a[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s_a, buffer_a_a, buffer_v_target, buffer_s_p, buffer_a_p, buffer_p_l = np.vstack(buffer_s_a), \
                                                                                                  np.vstack(buffer_a_a), \
                                                                                                  np.vstack(
                                                                                                      buffer_v_target), np.vstack(
                        buffer_s_p), np.vstack(buffer_a_p), np.vstack(buffer_p_l)
                    feed_dict = {
                        self.adv_net.s: buffer_s_a,
                        self.adv_net.a_his: buffer_a_a,
                        self.adv_net.v_target: buffer_v_target,
                        self.adv_net.x: buffer_s_p,
                        self.adv_net.y: buffer_a_p,
                        self.adv_net.p_labels: buffer_p_l,
                        self.adv_net.state_in[0]: batch_rnn_state[0],
                        self.adv_net.state_in[1]: batch_rnn_state[1]

                    }
                    # get new LSTM states
                    batch_rnn_state = sess.run(self.adv_net.state_out,
                                               feed_dict=feed_dict)
                    feed_dict = {
                        self.adv_net.s: buffer_s_a,
                        self.adv_net.a_his: buffer_a_a,
                        self.adv_net.v_target: buffer_v_target,
                        self.adv_net.x: buffer_s_p,
                        self.adv_net.y: buffer_a_p,
                        self.adv_net.p_labels: buffer_p_l,
                        self.adv_net.state_in[0]: batch_rnn_state[0],
                        self.adv_net.state_in[1]: batch_rnn_state[1]

                    }

                    self.adv_net.update_global(feed_dict)  # actual training step, update global ACNet
                    # adv_net.update_crit(feed_dict)
                    self.adv_net.train_prot(feed_dict)
                    self.adv_net.pull_prot()
                    buffer_s_p, buffer_s_a, buffer_a_p, buffer_a_a, buffer_r_a, buffer_p_l = [], [], [], [], [], []

                # update state variables
                t = t_
                v = v_
                a = a_
                x = x_
                v_rel = v_rel_
                x_rel = x_rel_
                t_h = t_h_
                x_lead = x_lead_
                v_lead = v_lead_
                a_lead = a_lead_
                prev_output = output
                total_step += 1
                # pythonapi.ApoClnt_PollAndSleep()  # poll client every now and then

            # end of episode training:
            if len(buffer_r_a) > 10:
                if t == 300 or crash == 1 or too_far == 1:
                    v_s_ = 0  # terminal state
                else:
                    v_s_ = sess.run(self.adv_net.v, {self.adv_net.s: np.reshape(s_a, (1, N_S)),
                                                     self.adv_net.x: np.reshape(s_p, (1, N_S_P))})[0, 0]
                buffer_v_target = []
                for r in buffer_r_a[::-1]:  # reverse buffer r
                    v_s_ = r + GAMMA * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()

                buffer_s_a, buffer_a_a, buffer_v_target, buffer_s_p, buffer_a_p, buffer_p_l = np.vstack(buffer_s_a), \
                                                                                              np.vstack(buffer_a_a), \
                                                                                              np.vstack(
                                                                                                  buffer_v_target), np.vstack(
                    buffer_s_p), np.vstack(buffer_a_p), np.vstack(buffer_p_l)
                feed_dict = {
                    self.adv_net.s: buffer_s_a,
                    self.adv_net.a_his: buffer_a_a,
                    self.adv_net.v_target: buffer_v_target,
                    self.adv_net.x: buffer_s_p,
                    self.adv_net.y: buffer_a_p,
                    self.adv_net.p_labels: buffer_p_l,
                    self.adv_net.state_in[0]: batch_rnn_state[0],
                    self.adv_net.state_in[1]: batch_rnn_state[1]

                }
                # get new LSTM states
                batch_rnn_state = sess.run(self.adv_net.state_out,
                                           feed_dict=feed_dict)
                feed_dict = {
                    self.adv_net.s: buffer_s_a,
                    self.adv_net.a_his: buffer_a_a,
                    self.adv_net.v_target: buffer_v_target,
                    self.adv_net.x: buffer_s_p,
                    self.adv_net.y: buffer_a_p,
                    self.adv_net.p_labels: buffer_p_l,
                    self.adv_net.state_in[0]: batch_rnn_state[0],
                    self.adv_net.state_in[1]: batch_rnn_state[1]

                }

                self.adv_net.update_global(feed_dict)  # actual training step, update global ACNet
                # adv_net.update_crit(feed_dict)
                self.adv_net.train_prot(feed_dict)
                self.adv_net.pull_prot()
                buffer_s_p, buffer_s_a, buffer_a_p, buffer_a_a, buffer_r_a, buffer_p_l = [], [], [], [], [], []

            # update tensorboard summaries
            rewards.append(ep_r)

            if self.name == 'W_0':  # track losses and actions of worker #0
                summary = sess.run(tf.summary.merge([sum_ploss, sum_vloss, sum_mu, sum_sigma, sum_v, sum_vt, sum_a,
                                                     sum_ap]), feed_dict=feed_dict)
                self.summary_writer.add_summary(summary, episodes)
                self.summary_writer.flush()

            summary = tf.Summary()

            summary.value.add(tag='Perf/Reward', simple_value=float(ep_r))
            summary.value.add(tag='Perf/Mean_Reward', simple_value=float(np.mean(arr_rewards)))
            summary.value.add(tag='Perf/Mean_Th', simple_value=float(np.mean(arr_th)))
            summary.value.add(tag='Perf/Min_Th', simple_value=float(np.min(arr_th)))
            self.summary_writer.add_summary(summary, episodes)
            self.summary_writer.flush()

            # print summary
            print(
                self.name,
                "Ep:", episodes,
                "| Ep_r: %i" % rewards[-1],
                "| Avg. Reward: %.5f" % np.mean(arr_rewards),
                "| Min. Reward: %.5f" % np.min(arr_rewards),
                "| Max. Reward: %.5f" % np.max(arr_rewards),
                "| Avg. Timeheadway: %.5f" % np.mean(arr_th),
                "| Min. Timeheadway: %.5f" % np.min(arr_th),
            )
            episodes += 1
            print('steps: %i' % b)

            # store eps with crashes
            if crash == 1:
                if not os.path.exists(args.fpath + args.log_dir + '/results'):
                    os.makedirs(args.fpath + args.log_dir + '/results')
                # calculate jerk array
                for k in range(0, 5):
                    arr_j.append(float(0))

                for k in range(5, len(arr_t)):
                    # calculate vehicle jerk
                    if abs(arr_t[k] - arr_t[k - 5]) != 0:
                        arr_j.append(((arr_a[k]) - (arr_a[k - 5])) / (arr_t[k] - arr_t[k - 5]))  # jerk
                    else:
                        arr_j.append(0)

                # write results to file
                headers = ['t', 'j', 'v', 'a', 'v_lead', 'a_lead', 'x_rel', 'v_rel', 'th', 'y_0', 'y_sc', 'sc', 'cof']
                with open(args.fpath + args.log_dir + '/results/' + str(episodes) + '.csv', 'w', newline='\n') as f:
                    wr = csv.writer(f, delimiter=',')
                    rows = zip(arr_t, arr_j, arr_v, arr_a, arr_v_leader, arr_a_leader, arr_x, arr_dv, arr_th,
                               arr_y_0,
                               arr_y_sc, arr_sc, arr_cof)
                    wr.writerow(headers)
                    wr.writerows(rows)


if __name__ == "__main__":
    rewards = []
    episodes = 0
    crashes = 0

    args = get_arguments()
    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)

    # define additional networks
    proxy = ipg_proxy.IpgProxy()  # Define proxy environment
    sl_net = sl_network.SupervisedNetwork()  # Define sl model

    # Create global network and local networks
    with graph.as_default():
        global_ac = ModelAdv(GLOBAL_NET_SCOPE, sess)
        workers = []
        for i in range(args.num_advs):
            w_name = 'W_%i' % i  # worker name
            workers.append(Worker(args, w_name, global_ac, sess, proxy, sl_net))  # create worker

    # tensorboard summaries
    i = 0  # track first env for tensorboard
    tf.summary.scalar('W_%i/loss/policy_loss' % i, workers[i].adv_net.a_loss)
    tf.summary.scalar('W_%i/loss/value_loss' % i, workers[i].adv_net.c_loss)
    tf.summary.histogram('W_%i/mu' % i, workers[i].adv_net.mu)
    tf.summary.histogram('W_%i/sigma' % i, workers[i].adv_net.sigma)
    tf.summary.histogram('W_%i/v' % i, workers[i].adv_net.v)
    tf.summary.histogram('W_%i/v_target' % i, workers[i].adv_net.v_target)
    tf.summary.histogram('W_%i/act_out' % i, workers[i].adv_net.A)

    with sess.as_default():
        with graph.as_default():
            saver = tf.train.Saver()
            tf.global_variables_initializer().run()
            coord = tf.train.Coordinator()

    worker_threads = []
    for worker in workers:  # start workers
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)

    print('Training finished!')
    print('Number of crashes: %d' % crashes)

    # save weights
    if not os.path.exists(args.fpath + args.log_dir):
        os.makedirs(args.fpath + args.log_dir)
    checkpoint_path = os.path.join(args.fpath + args.log_dir,
                                   "model-ep-%d-finalr-%d.ckpt" % (episodes, rewards[-1]))
    filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" % filename)
