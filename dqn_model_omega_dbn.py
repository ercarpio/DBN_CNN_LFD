# dqn_model_omega_dbn.py
# Estuardo Carpio-Mazariegos
# 10/24/2017

# helper function for printing tensor size
from __future__ import print_function

import tensorflow as tf
import numpy as np
import ConfigParser

# contains information relating to input data size
from constants import *

# inception network
from models.slim.nets.inception_resnet_v2 import inception_resnet_v2

CHECKPOINT = "inception_resnet_v2_2016_08_30.ckpt"
slim = tf.contrib.slim

# network layer information for P_CNN
layer_elements = [-1, 16, 32, 128, 3]

output_sizes = [32, 16, 4]
filter_sizes = [4, 4, 8]
stride_sizes = [2, 2, 4]
padding_size = [1, 1, 2]

# network layer information for A_CNN
aud_layer_elements = [-1, 16, 32, 128, 3]

aud_output_sizes = [(32, 6), (16, 4), (4, 4)]
aud_filter_sizes = [(8, 3), (4, 3), (8, 3)]
aud_stride_sizes = [(4, 1), (2, 1), (4, 1)]
aud_padding_size = [(2, 0), (1, 0), (2, 1)]

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
    validating - indicates if the model is being validated (True) or trained (False)
    """
    def __init__(self, graph_build=[1] * TOTAL_PARAMS, batch_size=1, filename="",
                 name="dqn", learning_rate=1e-5, log_dir="", validating=False):
        self.graph_build = graph_build
        self.__batch_size = batch_size
        self.__name = name
        self.__alpha = learning_rate
        self.log_dir = log_dir

        # LOADS CONFIGURATION
        config = ConfigParser.ConfigParser()
        config.read("config")
        self.gpu = config.get("MODEL", "GPU")

        # Model variables
        # noinspection PyCompatibility
        def weight_variable(name, shape):
            # initial = tf.truncated_normal(shape, stddev=0.1)
            initial = tf.truncated_normal(shape)
            initial = tf.div(initial, tf.sqrt(float(reduce(lambda a, b: a * b, shape))))
            return tf.Variable(initial, name=name)

        def bias_variable(name, shape):
            initial = tf.constant(0., shape=shape)
            return tf.Variable(initial, name=name)

        # Q variables
        self.variables_pnt = {
            "W1": weight_variable("W_conv1_pnt", [filter_sizes[0], filter_sizes[0],
                                                  pnt_dtype["num_c"], layer_elements[1]]),
            "b1": bias_variable("b_conv1_pnt", [layer_elements[1]]),
            "W2": weight_variable("W_conv2_pnt", [filter_sizes[1], filter_sizes[1],
                                                  layer_elements[1], layer_elements[2]]),
            "b2": bias_variable("b_conv2_pnt", [layer_elements[2]]),
            "W3": weight_variable("W_conv3_pnt", [filter_sizes[2], filter_sizes[2],
                                                  layer_elements[2], layer_elements[-2]]),
            "b3": bias_variable("b_conv3_pnt", [layer_elements[-2]])
        }

        self.variables_aud = {
            "W1": weight_variable("W_conv1_aud", [aud_filter_sizes[0][0],
                                                  aud_filter_sizes[0][1], aud_dtype["num_c"],
                                                  aud_layer_elements[1]]),
            "b1": bias_variable("b_conv1_aud", [aud_layer_elements[1]]),
            "W2": weight_variable("W_conv2_aud", [aud_filter_sizes[1][0],
                                                  aud_filter_sizes[1][1], aud_layer_elements[1],
                                                  aud_layer_elements[2]]),
            "b2": bias_variable("b_conv2_aud", [aud_layer_elements[2]]),
            "W3": weight_variable("W_conv3_aud", [aud_filter_sizes[2][0],
                                                  aud_filter_sizes[2][1], aud_layer_elements[2],
                                                  aud_layer_elements[3]]),
            "b3": bias_variable("b_conv3_aud", [aud_layer_elements[3]])
        }

        self.variables_lstm = {
            "W_lstm": weight_variable("W_lstm", [layer_elements[-2], layer_elements[-1]]),
            "b_lstm": bias_variable("b_lstm", [layer_elements[-1]]),
            "W_fc": weight_variable("W_fc", [layer_elements[-1] * 2, layer_elements[-1]]),
            "b_fc": bias_variable("b_fc", [layer_elements[-1]])
        }

        # Q^hat variables
        self.variables_pnt_hat = {
            "W1": weight_variable("W_conv1_pnt_hat", [filter_sizes[0], filter_sizes[0],
                                                      pnt_dtype["num_c"], layer_elements[1]]),
            "b1": bias_variable("b_conv1_pnt_hat", [layer_elements[1]]),
            "W2": weight_variable("W_conv2_pnt_hat", [filter_sizes[1], filter_sizes[1],
                                                      layer_elements[1], layer_elements[2]]),
            "b2": bias_variable("b_conv2_pnt_hat", [layer_elements[2]]),
            "W3": weight_variable("W_conv3_pnt_hat", [filter_sizes[2], filter_sizes[2],
                                                      layer_elements[2], layer_elements[-2]]),
            "b3": bias_variable("b_conv3_pnt_hat", [layer_elements[-2]])
        }

        self.variables_aud_hat = {
            "W1": weight_variable("W_conv1_aud_hat", [aud_filter_sizes[0][0],
                                                      aud_filter_sizes[0][1], aud_dtype["num_c"],
                                                      aud_layer_elements[1]]),
            "b1": bias_variable("b_conv1_aud_hat", [aud_layer_elements[1]]),
            "W2": weight_variable("W_conv2_aud_hat", [aud_filter_sizes[1][0],
                                                      aud_filter_sizes[1][1], aud_layer_elements[1],
                                                      aud_layer_elements[2]]),
            "b2": bias_variable("b_conv2_aud_hat", [aud_layer_elements[2]]),
            "W3": weight_variable("W_conv3_aud_hat", [aud_filter_sizes[2][0],
                                                      aud_filter_sizes[2][1], aud_layer_elements[2],
                                                      aud_layer_elements[3]]),
            "b3": bias_variable("b_conv3_aud_hat", [aud_layer_elements[3]])
        }

        self.variables_lstm_hat = {
            "W_lstm": weight_variable("W_lstm_hat", [layer_elements[-2], layer_elements[-1]]),
            "b_lstm": bias_variable("b_lstm_hat", [layer_elements[-1]]),
            "W_fc": weight_variable("W_fc_hat", [layer_elements[-1] * 2, layer_elements[-1]]),
            "b_fc": bias_variable("b_fc_hat", [layer_elements[-1]])
        }

        # Placeholder variables
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

        # placeholder for how many prompts have been delivered
        self.temporal_info_ph = tf.placeholder("float32", [layer_elements[-1]],
                                               name="temporal_info_placeholder")

        # placeholder for the reward values to classify with
        self.y_ph = tf.placeholder("float", [None, layer_elements[-1]], name="y_placeholder")

        # Model functions
        self.pred_var_set = self.execute_model_DQN_var_set()  # used to initialize variables
        self.pred = self.execute_model_DQN()  # used for training
        self.pred_hat = self.execute_model_DQN_hat()  # used to evaluate q_hat
        self.max_q_hat = tf.reduce_max(self.pred_hat, axis=1)

        # inception variables
        self.variables_img = {}
        exclusions = ["InceptionResnetV2/Logits", "InceptionResnetV2/AuxLogits"]

        if self.graph_build[0]:
            slim.assign_from_checkpoint_fn(CHECKPOINT, slim.get_model_variables())

        variables_to_train = [x for x in tf.trainable_variables()
                              if x.op.name in ["W_fc", "b_fc", "W_fc_hat", "b_fc_hat"]]

        self.variables_img_main = {}
        self.variables_img_hat = {}
        for k in self.variables_img.keys():
            self.variables_img_main[k] = tf.Variable(self.variables_img[k].initialized_value())
            self.variables_img_hat[k] = tf.Variable(self.variables_img[k].initialized_value())

        self.diff = self.y_ph - tf.clip_by_value(self.pred, 1e-10, 100)

        self.cross_entropy = tf.reduce_mean(tf.square(self.diff))
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.__alpha).minimize(
            self.cross_entropy, var_list=variables_to_train)

        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y_ph, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        # session
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.loader = tf.train.Saver(var_list=[x for x in tf.trainable_variables()
                                              if x not in variables_to_train])
        self.saver = tf.train.Saver()

        if validating:
            self.loader = self.saver

        self.merged_summary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.log_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.log_dir + '/test')
        self.graph_writer = tf.summary.FileWriter(self.log_dir + '/projector', self.sess.graph)

        if len(filename) == 0:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)  # remove when using a saved file
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            print("RESTORING VALUES")
            self.loader.restore(self.sess, filename)

    def save_model(self, name="model.ckpt", save_dir=""):
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

    def assign_variables(self):
        # for the inception network variables
        self.restore_q_hat_vars(self.variables_img_main, self.variables_img_hat)

        # for all other network variables
        self.restore_q_hat_vars(self.variables_pnt, self.variables_pnt_hat)
        self.restore_q_hat_vars(self.variables_aud, self.variables_aud_hat)
        self.restore_q_hat_vars(self.variables_lstm, self.variables_lstm_hat)

    def gen_prediction(self, num_frames, img_data, pnt_data, aud_data, num_prompts):
        # used by the ASD robot
        partitions = np.zeros((1, num_frames))
        print("partitions.shape: ", partitions.shape)
        partitions[0][-1] = 1
        print("num_prompts: ", num_prompts)
        with tf.variable_scope(self.__name) as scope:
            prediction = self.sess.run(self.pred, feed_dict={  # generate_pred
                self.seq_length_ph: [num_frames],
                self.img_ph: img_data,
                self.pnt_ph: pnt_data,
                self.aud_ph: aud_data,
                self.partitions_ph: partitions,
                self.train_ph: False,
                self.temporal_info_ph: num_prompts
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
                          "",
                          self.variables_pnt,
                          self.variables_aud,
                          self.variables_lstm,
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
                          "",
                          self.variables_pnt,
                          self.variables_aud,
                          self.variables_lstm
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
                          "",
                          self.variables_pnt_hat,
                          self.variables_aud_hat,
                          self.variables_lstm_hat
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

    # THE MODEL
    def model(self, seq_length, img_ph, pnt_ph, aud_ph, partitions_ph, train_ph,
              prompts_ph, variable_scope, variable_scope2, var_img, var_pnt,
              var_aud, var_lstm, incep_reuse=True):  #

        def process_vars(seq, data_type):
            # cast inputs to the correct data type
            seq_inp = tf.cast(seq, tf.float32)
            return tf.reshape(seq_inp, (self.__batch_size, -1, data_type["cmp_h"],
                                        data_type["cmp_w"], data_type["num_c"]))

        def convolve_data_inception(input_data, val, n, dtype):
            data = tf.reshape(input_data, [-1, 299, 299, 3])
            logits, end_points = inception_resnet_v2(data,
                                                     num_classes=output_sizes[-1] *
                                                                 output_sizes[-1] *
                                                                 layer_elements[-2],
                                                     is_training=False, reuse=incep_reuse)
            return logits

        def check_legal_inputs(tensor, name):
            return tf.verify_tensor_all_finite(tensor, "ERR: Tensor not finite - " + name,
                                               name=name)

        def convolve_data_3layer_pnt(input_data, val, variables, n, dtype):
            def pad_tf(x, p):
                return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT")

            def gen_convolved_output(sequence, W, b, stride, num_hidden,
                                     new_size, train_ph, padding='SAME'):
                conv = tf.nn.conv2d(sequence, W, strides=[1, stride, stride, 1],
                                    padding=padding) + b
                return tf.nn.relu(conv)

            input_data = tf.reshape(input_data, [-1, dtype["cmp_h"], dtype["cmp_w"],
                                                 dtype["num_c"]], name=n + "_inp_reshape")

            input_data = pad_tf(input_data, padding_size[0])
            padding = "VALID"

            input_data = gen_convolved_output(input_data, variables["W1"], variables["b1"],
                                              stride_sizes[0], layer_elements[1],
                                              output_sizes[0], train_ph, padding)
            self.variable_summaries(input_data, dtype["name"] + "_conv1")
            input_data = tf.verify_tensor_all_finite(input_data, "ERR: Tensor not finite - ",
                                                     name="conv1_" + n)

            input_data = pad_tf(input_data, padding_size[1])
            padding = "VALID"

            input_data = gen_convolved_output(input_data, variables["W2"], variables["b2"],
                                              stride_sizes[1], layer_elements[2],
                                              output_sizes[1], train_ph, padding)
            self.variable_summaries(input_data, dtype["name"] + "_conv2")
            input_data = tf.verify_tensor_all_finite(input_data, "ERR: Tensor not finite - ",
                                                     name="conv2_" + n)

            input_data = pad_tf(input_data, padding_size[2])
            padding = "VALID"

            input_data = gen_convolved_output(input_data, variables["W3"], variables["b3"],
                                              stride_sizes[-1], layer_elements[-2],
                                              output_sizes[-1], train_ph, padding)
            self.variable_summaries(input_data, dtype["name"] + "_conv3")
            input_data = tf.verify_tensor_all_finite(input_data, "ERR: Tensor not finite - ",
                                                     name="conv3_" + n)
            return input_data

        def convolve_data_3layer_aud(input_data, val, variables, n, dtype):
            def pad_tf(x, padding):
                return tf.pad(x, [[0, 0], [padding[0], padding[0]],
                                  [padding[1], padding[1]], [0, 0]], "CONSTANT")

            def gen_convolved_output(sequence, W, b, stride, num_hidden,
                                     new_size, train_ph, padding='SAME'):
                conv = tf.nn.conv2d(sequence, W, strides=[1, stride[0], stride[1], 1],
                                    padding=padding) + b
                return tf.nn.relu(conv)

            input_data = tf.reshape(input_data, [-1, dtype["cmp_h"], dtype["cmp_w"],
                                                 dtype["num_c"]], name=n + "_inp_reshape")

            input_data = pad_tf(input_data, aud_padding_size[0])
            padding = "VALID"

            input_data = gen_convolved_output(input_data, variables["W1"], variables["b1"],
                                              aud_stride_sizes[0], aud_layer_elements[1],
                                              aud_output_sizes[0], train_ph, padding)
            self.variable_summaries(input_data, dtype["name"] + "_conv1")
            input_data = tf.verify_tensor_all_finite(input_data,
                                                     "ERR: Tensor not finite - conv1_" + n,
                                                     name="conv1_" + n)

            input_data = pad_tf(input_data, aud_padding_size[1])
            padding = "VALID"

            input_data = gen_convolved_output(input_data, variables["W2"], variables["b2"],
                                              aud_stride_sizes[1], aud_layer_elements[2],
                                              aud_output_sizes[1], train_ph, padding)
            self.variable_summaries(input_data, dtype["name"] + "_conv2")
            input_data = tf.verify_tensor_all_finite(input_data,
                                                     "ERR: Tensor not finite - conv2_" + n,
                                                     name="conv2_" + n)

            input_data = pad_tf(input_data, aud_padding_size[2])
            padding = "VALID"

            input_data = gen_convolved_output(input_data, variables["W3"], variables["b3"],
                                              aud_stride_sizes[2], aud_layer_elements[3],
                                              aud_output_sizes[2], train_ph, padding)
            self.variable_summaries(input_data, dtype["name"] + "_conv3")
            input_data = tf.verify_tensor_all_finite(input_data,
                                                     "ERR: Tensor not finite - conv3_" + n,
                                                     name="conv3_" + n)
            return input_data

        # CNN Stacks
        # Storage Variables
        # 0 - RGB, 1 - Optical Flow, 2 - Audio
        inp_data = [0] * TOTAL_PARAMS
        conv_inp = [0] * TOTAL_PARAMS

        with tf.device(self.gpu):
            # Inception Network (INRV2)
            if self.graph_build[0]:
                val = 0
                inp_data[val] = process_vars(img_ph, img_dtype)
                conv_inp[val] = convolve_data_inception(inp_data[val], val, "img", img_dtype)

            with variable_scope as scope:
                # P_CNN
                if self.graph_build[1]:
                    val = 1
                    inp_data[val] = process_vars(pnt_ph, pnt_dtype)
                    conv_inp[val] = convolve_data_3layer_pnt(inp_data[val], val, var_pnt,
                                                             "pnt", pnt_dtype)

                # A_CNN
                if self.graph_build[2]:
                    val = 2
                    inp_data[val] = process_vars(aud_ph, aud_dtype)
                    conv_inp[val] = convolve_data_3layer_aud(inp_data[val], val, var_aud,
                                                             "aud", aud_dtype)

                # Combine Output of CNN Stacks
                combined_data = None
                for i in range(TOTAL_PARAMS):
                    if self.graph_build[i]:
                        if i < 2:
                            conv_inp[i] = tf.reshape(conv_inp[i],
                                                     [self.__batch_size, -1,
                                                      output_sizes[-1] *
                                                      output_sizes[-1] *
                                                      layer_elements[-2]],
                                                     name="combine_reshape")
                        else:
                            conv_inp[i] = tf.reshape(conv_inp[i],
                                                     [self.__batch_size, -1,
                                                      aud_output_sizes[-1][0] *
                                                      aud_output_sizes[-1][0] *
                                                      aud_layer_elements[-2]],
                                                     name="combine_reshape_aud")

                        if combined_data is None:
                            combined_data = conv_inp[i]
                        else:
                            combined_data = tf.concat([combined_data, conv_inp[i]], 2)

                        combined_data = check_legal_inputs(combined_data, "combined_data")

                # capture variables before changing scope
                W_lstm = var_lstm["W_lstm"]
                b_lstm = var_lstm["b_lstm"]
                W_fc = var_lstm["W_fc"]
                b_fc = var_lstm["b_fc"]

            with variable_scope2 as scope:
                # Internal Temporal Information (LSTM)
                lstm_cell = tf.contrib.rnn.LSTMCell(layer_elements[-2], use_peepholes=False,
                                                    cell_clip=None, initializer=None,
                                                    num_proj=None, proj_clip=None,
                                                    forget_bias=1.0, state_is_tuple=True,
                                                    activation=None, reuse=None)

                lstm_mat, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=combined_data,
                                                dtype=tf.float32, sequence_length=seq_length,
                                                time_major=False)

                # if lstm_out is NaN replace with 0 to prevent model breakage
                lstm_mat = tf.where(tf.is_nan(lstm_mat), tf.zeros_like(lstm_mat), lstm_mat)
                lstm_mat = check_legal_inputs(lstm_mat, "lstm_mat")

                # extract relevant information from LSTM output using partitions
                num_partitions = 2
                lstm_out = tf.dynamic_partition(lstm_mat, partitions_ph, num_partitions)[1]

                # FC1
                fc1_out = tf.matmul(lstm_out, W_lstm) + b_lstm
                fc1_out = check_legal_inputs(fc1_out, "fc1")
                self.variable_summaries(fc1_out, "fc1")

                # External Temporal Information (using DBN)
                prompts_ph = tf.reshape(prompts_ph, [-1, layer_elements[-1]])
                fc1_prompt = tf.concat([fc1_out, prompts_ph], 1)

                # FC2: generate final q-values
                fc2_out = tf.matmul(fc1_prompt, W_fc) + b_fc
                fc2_out = check_legal_inputs(fc2_out, "fc2")
                self.variable_summaries(fc2_out, "fc")

                return fc2_out


if __name__ == '__main__':
    dqn = DQNModel([1, 0, 0])
