# dqn_model_omega_dbn.py
# Estuardo Carpio
# 10/23/2017

# helper function for printing tensor size
from __future__ import print_function

import tensorflow as tf
import numpy as np

# incpetion network
from models.slim.nets.inception_resnet_v2 import inception_resnet_v2

# contains information relating to input data size
from constants import *

CHECKPOINT = "inception_resnet_v2_2016_08_30.ckpt"
CHECKPOINT_NP = "np_omega/model.ckpt"
slim = tf.contrib.slim

# network layer information for P_CNN
layer_elements = [-1, 3]

# network layer information for A_CNN
aud_layer_elements = [-1, 3]

TOTAL_PARAMS = 3


class DQNModel:
    """
    graphbuild - bool array with len = TOTAL_PARAMS, indicates which CNNs should be active
                (all by default)
    batch_size - int (1 by default)
    filename - string, location of file with saved model parameters (no model listed by default)
    name - string, model name ("dqn" by default)
    learning_rate - float, speed at which the model trains (1e-5 by default)
    log_dir - directory to store summary information (no directory listed by default)
    """
    def __init__(self, graphbuild=[1] * TOTAL_PARAMS, batch_size=1, filename="",
                 name="dqn", learning_rate=1e-5, log_dir=""):
        self.graphbuild = graphbuild
        self.__batch_size = batch_size
        self.__name = name
        self.__alpha = learning_rate
        self.log_dir = log_dir

        # ---------------------------------------
        # Model variables
        # ---------------------------------------
        def weight_variable(name, shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(name, shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        # Q variables
        self.variables_fc = {
            "W_fc": weight_variable("W_fc", [layer_elements[-1] * 2, layer_elements[-1]]),
            "b_fc": bias_variable("b_fc", [layer_elements[-1]])
        }

        # Q-hat variables
        self.variables_fc_hat = {
            "W_fc": weight_variable("W_fc_hat", [layer_elements[-1] * 2, layer_elements[-1]]),
            "b_fc": bias_variable("b_fc_hat", [layer_elements[-1]])
        }

        # ---------------------------------------
        # Placeholder variables
        # ---------------------------------------
        # placeholder for the RGB data
        self.img_ph = tf.placeholder("float",
                                     [self.__batch_size, None,
                                      img_dtype["cmp_h"] * img_dtype["cmp_w"] * img_dtype["num_c"]],
                                     name="img_placeholder")

        # placeholder for the Optical Flow data
        self.pnt_ph = tf.placeholder("float",
                                     [self.__batch_size, None,
                                      pnt_dtype["cmp_h"] * pnt_dtype["cmp_w"] * pnt_dtype["num_c"]],
                                     name="pnt_placeholder")

        # placeholder for the Audio data
        self.aud_ph = tf.placeholder("float",
                                     [self.__batch_size, None,
                                      aud_dtype["cmp_h"] * aud_dtype["cmp_w"] * aud_dtype["num_c"]],
                                     name="aud_placeholder")

        # placeholder for the sequence length
        self.seq_length_ph = tf.placeholder("int32", [self.__batch_size],
                                            name="seq_len_placeholder")

        # placeholder for where each sequence ends in a matrix
        self.partitions_ph = tf.placeholder("int32", [self.__batch_size, None],
                                            name="partition_placeholder")

        # placeholder for boolean listing whether the network is being trained or evaluated
        self.train_ph = tf.placeholder("bool", [], name="train_placeholder")

        # placeholder for temporal information
        self.temporal_info_ph = tf.placeholder("float32", [layer_elements[-1]],
                                               name="temporal_info_placeholder")

        # placeholder for the reward values to classify with
        self.y_ph = tf.placeholder("float", [None, layer_elements[-1]], name="y_placeholder")

        # ---------------------------------------
        # Model functions
        # ---------------------------------------
        self.pred_var_set = self.execute_model_DQN_var_set()  # used to initialize variables
        self.pred = self.execute_model_DQN()  # used for training
        self.pred_hat = self.execute_model_DQN_hat()  # used to evaluate q_hat
        self.max_q_hat = tf.reduce_max(self.pred_hat, axis=1)
        self.pred_index = tf.argmax(self.pred, 1)
        # inception variables
        self.variables_img = {}
        variables_to_restore = []
        if (self.graphbuild[0]):
            vars = []
            for var in slim.get_model_variables(CHECKPOINT):
                vars.append(var)
            slim.assign_from_checkpoint_fn(CHECKPOINT, vars)
            variables_to_restore.append(vars)
            vars = []
            for var in slim.get_model_variables(CHECKPOINT_NP):
                vars.append(var)
            slim.assign_from_checkpoint_fn(CHECKPOINT_NP, vars)
            variables_to_restore.append(vars)

        variables_to_train = [x for x in tf.trainable_variables() if x not in variables_to_restore]

        self.variables_img_main = {}
        self.variables_img_hat = {}
        for k in self.variables_img.keys():
            self.variables_img_main[k] = tf.Variable(self.variables_img[k].initialized_value())
            self.variables_img_hat[k] = tf.Variable(self.variables_img[k].initialized_value())

        self.diff = self.y_ph - tf.clip_by_value(self.pred, 1e-10, 100)

        self.cross_entropy = tf.reduce_mean(tf.square(self.diff))
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.__alpha).minimize(self.cross_entropy,
                                                                                     var_list=variables_to_train)
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_ph, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)
        self.predict_output = tf.argmax(self.pred, 1)

        # session
        self.sess = tf.InteractiveSession()

        self.saver = tf.train.Saver()

        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')
        self.graph_writer = tf.summary.FileWriter(self.log_dir + '/projector', self.sess.graph)

        if (len(filename) == 0):
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)  # remove when using a saved file
        else:
            print("RESTORING VALUES")
            self.saver.restore(self.sess, filename)

    def saveModel(self, name="model.ckpt", save_dir=""):
        self.saver.save(self.sess, save_dir + '/' + name)

    def restore_q_hat_vars(self, src, dst):
        arr = []
        var_dst = []
        for k in dst.keys():
            arr.append(src[k])
            var_dst.append(dst[k])
        arr = self.sess.run(arr)
        for seq, var in zip(arr, var_dst):
            v = np.array(seq).reshape(np.array(seq).shape)
            var.load(v, session=self.sess)

    def assignVariables(self):
        # for the inception network variables
        self.restore_q_hat_vars(self.variables_img_main, self.variables_img_hat)

    def genPrediction(self, num_frames, img_data, pnt_data, aud_data, temporal_info):
        # used by the ASD robot
        partitions = np.zeros((1, num_frames))
        print("partitions.shape: ", partitions.shape)
        partitions[0][-1] = 1
        print("temporal_info: ", temporal_info)
        with tf.variable_scope(self.__name) as scope:
            prediction = self.sess.run(self.pred, feed_dict={  # generate_pred
                self.seq_length_ph: [num_frames],
                self.img_ph: img_data,
                self.pnt_ph: pnt_data,
                self.aud_ph: aud_data,
                self.partitions_ph: partitions,
                self.train_ph: False,
                self.temporal_info_ph: temporal_info
            })
            print(prediction, np.max(prediction), np.argmax(prediction))
            return np.argmax(prediction)  # prediction[0]

    def execute_model_DQN_var_set(self):
        return self.model(self.seq_length_ph,
                          self.img_ph,
                          self.pnt_ph,
                          self.aud_ph,
                          self.partitions_ph,
                          self.train_ph,
                          self.temporal_info_ph,
                          tf.variable_scope("dqn"),
                          tf.variable_scope("dqn"),
                          self.variables_fc,
                          False
                          )

    def execute_model_DQN(self):
        return self.model(self.seq_length_ph,
                          self.img_ph,
                          self.pnt_ph,
                          self.aud_ph,
                          self.partitions_ph,
                          self.train_ph,
                          self.temporal_info_ph,
                          tf.variable_scope("dqn"),
                          tf.variable_scope("dqn", reuse=True),
                          self.variables_fc
                          )

    def execute_model_DQN_hat(self):
        return self.model(self.seq_length_ph,
                          self.img_ph,
                          self.pnt_ph,
                          self.aud_ph,
                          self.partitions_ph,
                          self.train_ph,
                          self.temporal_info_ph,
                          tf.variable_scope("dqn_hat"),
                          tf.variable_scope("dqn", reuse=True),
                          self.variables_fc_hat
                          )

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries' + name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

            #####################
            ##### THE MODEL #####
            #####################

    def model(self, seq_length, img_ph, pnt_ph, aud_ph, partitions_ph, train_ph, prompts_ph, variable_scope,
              variable_scope2, var_fc, incep_reuse=True):  #

        def process_vars(seq, data_type):
            # cast inputs to the correct data type
            seq_inp = tf.cast(seq, tf.float32)
            return tf.reshape(seq_inp,
                              (self.__batch_size, -1, data_type["cmp_h"], data_type["cmp_w"], data_type["num_c"]))

        def check_legal_inputs(tensor, name):
            return tf.verify_tensor_all_finite(tensor, "ERR: Tensor not finite - " + name,
                                               name=name)

        W_fc = var_fc["W_fc"]
        b_fc = var_fc["b_fc"]

        with variable_scope2 as scope:
            # ---------------------------------------
            # External Temporal Information
            # ---------------------------------------
            prompts_ph = tf.reshape(prompts_ph, [-1, layer_elements[-1]])
            fc1_prompt = tf.concat([fc1_out, prompts_ph], 1)

            # FC2: generate final q-values
            fc2_out = tf.matmul(fc1_prompt, W_fc) + b_fc
            fc2_out = check_legal_inputs(fc2_out, "fc2")
            self.variable_summaries(fc2_out, "fc")

            return fc2_out


if __name__ == '__main__':
    dqn = DQNModel([1, 0, 0])
