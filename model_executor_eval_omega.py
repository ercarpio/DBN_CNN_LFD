# model_trainer_omega.py
# Madison Clark-Turner
# 10/13/2017

from __future__ import print_function

# DQN MODEL
from dqn_model_omega_dbn import *
from input_pipeline import *

import tensorflow as tf
import numpy as np

# HELPER FUNCTIONS
import sys
import os
from os.path import isfile, join
from datetime import datetime

# DBN
from dbn_cnn_interface import DbnCnnInterface

if __name__ == '__main__':
    # Input params
    print("time start: ", str(datetime.now()))

    # LOAD CONFIGURATION FILE
    config = ConfigParser.ConfigParser()
    config.read("config")
    alpha = config.getfloat("EVALUATION", "ALPHA")
    num_iter = config.getint("EVALUATION", "NUM_ITER")
    tfrecords_path = config.get("EVALUATION", "TFRECORDS_PATH")
    dqn_checkpoint = config.get("EVALUATION", "DQN_CHECKPOINT")

    graph_build = [1, 1, 1]

    # Read contents of TFRecord file
    file_names = [f for f in os.listdir(tfrecords_path) if isfile(join(tfrecords_path, f))]
    file_names = [tfrecords_path + x for x in file_names]
    file_names.sort()

    #  Prepare DBN
    dbn = DbnCnnInterface()

    # Generate Model
    dqn = DQNModel(graph_build, batch_size=BATCH_SIZE, learning_rate=alpha,
                   filename=dqn_checkpoint, log_dir="LOG_DIR", validating=True)

    # Train Model
    coord = tf.train.Coordinator()

    slen_t, slen_pr_t, i_t, p_t, a_t, pl_t, l_t, i_pr_t, p_pr_t, a_pr_t, name_t = \
        input_pipeline(file_names)
    l_t = tf.squeeze(l_t, [1])

    dqn.sess.run(tf.local_variables_initializer())
    threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

    # TESTING
    print("BEGIN TESTING")
    failed, good = [], []
    failed_names = []

    for iteration in range(num_iter):
        print("iteration: ", iteration)
        n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2, \
        pnt_data2, aud_data2, names = dqn.sess.run([slen_t, slen_pr_t, i_t, p_t, a_t, pl_t,
                                                    l_t, i_pr_t, p_pr_t, a_pr_t, name_t])

        print(n_seq, img_data.shape)
        partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))

        for x in range(BATCH_SIZE):
            if np.max(n_seq) > 0:
                v = n_seq[x] - 1
                if v < 0:
                    v = 0
                partitions_1[x][v] = 1

        # Filter possible actions using DBN
        dbn_output = dbn.filter_q_values([1, 1, 1], evidence=num_prompts[0])

        pred, equ = dqn.sess.run([dqn.pred, dqn.correct_pred], feed_dict={
            dqn.seq_length_ph: n_seq,
            dqn.img_ph: img_data,
            dqn.pnt_ph: pnt_data,
            dqn.aud_ph: aud_data,
            dqn.y_ph: label_data,
            dqn.partitions_ph: partitions_1,
            dqn.train_ph: False,
            dqn.temporal_info_ph: dbn_output})

        print("name:", names)
        print("pred:", pred)
        print("correct:", equ)

        for t in range(len(equ)):
            tup = [names[t], pred[t], label_data[t], num_prompts[0]]
            if not equ[t] and (names[t] not in failed_names):
                failed.append(tup)
                failed_names.append(names[t])
            elif equ[t] and (names[t] not in failed_names):
                good.append(tup)
                failed_names.append(names[t])

    print("failed:")
    for p in failed:
        print(str(p[0]), str(p[3]), str(p[1]), str(np.argmax(p[2])))

    print("good:")
    for p in good:
        print(str(p[0]), str(p[3]), str(p[1]), str(np.argmax(p[2])))

    bins = [[0, 0], [0, 0], [0, 0]]
    for p in good:
        bins[np.argmax(p[2])][0] += 1
    for p in failed:
        bins[np.argmax(p[2])][1] += 1

    print("bins", bins)
    print("Accuracy:", (len(good)) / float(len(good) + len(failed)))
    print("Prompt Accuracy: ", bins[0][0] / float(bins[0][0] + bins[0][1]))
    print("Reward Accuracy: ", bins[1][0] / float(bins[1][0] + bins[1][1]))
    print("Abort Accuracy: ", bins[2][0] / float(bins[2][0] + bins[2][1]))

    bins = {'z': [0, 0], 'g': [0, 0], 'a': [0, 0], 'zg': [0, 0],
            'ga': [0, 0], 'zg': [0, 0], 'zga': [0, 0], 'none': [0, 0]}
    for p in good:
        for k in bins:
            act = np.argmax(p[2])
            if p[0][5:].find(k) >= 0 and (act >= 1):
                bins[k][0] += 1
    for p in failed:
        for k in bins:
            act = np.argmax(p[2])
            if p[0][5:].find(k) >= 0 and (act >= 1):
                bins[k][1] += 1

    print("bins", bins)
    for k in bins:
        print(k + " Accuracy: ", bins[k][0] / float(bins[k][0] + bins[k][1]))

    print(len(good) + len(failed), len(failed_names))
    print(failed_names)
