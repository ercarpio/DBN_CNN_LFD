# model_trainer_omega.py
# Madison Clark-Turner
# 10/13/2017

# FILE IO
import ConfigParser
from input_pipeline import *
import os
from os.path import isfile, join

# HELPER METHODS
import sys
from datetime import datetime

# MODEL STRUCTURE
from dqn_model_omega_dbn import *

# DBN
from dbn_cnn_interface import DbnCnnInterface

if __name__ == '__main__':
    print("time start: ", str(datetime.now()))

    # Command-Line Parameters
    graph_build = [0] * TOTAL_PARAMS
    if len(sys.argv) > 1:
        if len(sys.argv[1]) == 1 and int(sys.argv[1]) < 3:
            graph_build[int(sys.argv[1])] = 1
        else:
            print("Usage: python model_trainer_omega.py <args>")
            print("\t0 - only build network with RGB information")
            print("\t1 - only build network with Optical Flow information")
            print("\t2 - only build network with Audio information")
            print("\t(nothing) - build network with all information")
    else:
        graph_build = [1] * TOTAL_PARAMS

    if sum(graph_build) < 3:
        print("BUILDING PARTIAL MODEL")

    # LOAD CONFIGURATION FILE
    config = ConfigParser.ConfigParser()
    config.read("config")
    gamma = config.getfloat("TRAINING", "GAMMA")
    alpha = config.getfloat("TRAINING", "ALPHA")
    num_iter = config.getint("TRAINING", "NUM_ITER")
    tfrecords_path = config.get("TRAINING", "TFRECORDS_PATH")
    dqn_checkpoint = config.get("TRAINING", "DQN_CHECKPOINT")
    metrics_freq = config.getint("TRAINING", "METRICS_FREQ")
    prediction_freq = config.getint("TRAINING", "PREDICTION_FREQ")
    updates_freq = config.getint("TRAINING", "UPDATES_FREQ")
    checkpoint_freq = config.getint("TRAINING", "CHECKPOINT_FREQ")

    # Read contents of TFRecord file
    # generate list of file names
    file_names = [f for f in os.listdir(tfrecords_path) if isfile(join(tfrecords_path, f))]
    file_names = [tfrecords_path + x for x in file_names]
    file_names.sort()

    # Prepare DBN
    dbn = DbnCnnInterface()

    # Generate Model
    dqn = DQNModel(graph_build, batch_size=BATCH_SIZE, learning_rate=alpha,
                   filename=dqn_checkpoint, log_dir="LOG_DIR")

    # if building from checkpoint need to setup dqn_hat variables
    if dqn_checkpoint != "":
        dqn.assignVariables()

    # Train Model
    coord = tf.train.Coordinator()
    '''
    sequence length - slen
    sequence length prime - slen_pr
    image raw - i
    points raw - p
    audio raw - a
    previous action - pl
    action - l
    image raw prime - i_pr
    points raw prime - p_pr
    audio raw prime - a_pr
    file identifier - n_id
    '''
    # read records from files into tensors
    slen, slen_pr, i, p, a, pl, l, i_pr, p_pr, a_pr, n_id = input_pipeline(file_names)
    l = tf.squeeze(l, [1])

    # initialize all variables
    dqn.sess.run(tf.local_variables_initializer())
    dqn.sess.graph.finalize()
    threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

    print("Num epochs: " + str(NUM_EPOCHS) + ", Batch Size: " + str(BATCH_SIZE) + ", Num Files: " +
          str(len(file_names)) + ", Num iterations: " + str(num_iter))

    for iteration in range(num_iter):
        # read a batch of tfrecords into np arrays
        n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, \
        img_data2, pnt_data2, aud_data2, name = dqn.sess.run([slen, slen_pr, i, p, a, pl, l,
                                                              i_pr, p_pr, a_pr, n_id])

        # generate partitions; used for extracting relevant data from the LSTM layer
        partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
        partitions_2 = np.zeros((BATCH_SIZE, np.max(n_seq2)))

        for x in range(BATCH_SIZE):
            if np.max(n_seq) > 0:
                v = n_seq[x] - 1
                if v < 0:
                    v = 0
                partitions_1[x][v] = 1
            if np.max(n_seq2) > 0:
                v = n_seq2[x] - 1
                if v < 0:
                    v = 0
                partitions_2[x][v] = 1

        # generate y_i for not terminal states
        newy = 0
        if np.max(n_seq2) > 1:
            # if at least on of the input files in the batch is not terminal then we need
            # to shape and pass the subsequent observation into the network in order to
            # generate a q-value from Q^hat
            img_data2 = set_shape(img_data2, img_dtype)
            pnt_data2 = set_shape(pnt_data2, pnt_dtype)
            aud_data2 = set_shape(aud_data2, aud_dtype)

            # Filter possible actions using DBN
            dbn_output = dbn.filter_q_values([1, 1, 1], evidence=np.sign(n_seq2))

            vals = {
                dqn.seq_length_ph: n_seq2,
                dqn.img_ph: img_data2,
                dqn.pnt_ph: pnt_data2,
                dqn.aud_ph: aud_data2,
                dqn.partitions_ph: partitions_2,
                dqn.train_ph: False,
                dqn.temporal_info_ph: dbn_output
            }

            # get the maximum q-value from q^hat
            newy = dqn.sess.run(dqn.max_q_hat, feed_dict=vals)
            # assign the max q-value to the appropriate action
            newy *= np.sign(n_seq2)
        else:
            newy = np.zeros(BATCH_SIZE)

        # set up array for y_i and populate appropriately
        r = np.array(label_data)

        # reward given for executing the prompt action
        # the reward for abort and reward actions is 1.0
        r[:, 0] = r[:, 0] * .2

        for j in range(r.shape[0]):
            for v in range(r.shape[1]):
                if r[j][v] != 0:
                    if v < 2:
                        r[j][v] += newy[j] * gamma

        # Filter possible actions using DBN
        dbn_output = dbn.filter_q_values([1, 1, 1], evidence=num_prompts[0])

        # Optimize Network
        vals = {
            dqn.seq_length_ph: n_seq,
            dqn.img_ph: img_data,
            dqn.pnt_ph: pnt_data,
            dqn.aud_ph: aud_data,
            dqn.y_ph: r,
            dqn.partitions_ph: partitions_1,
            dqn.train_ph: True,
            dqn.temporal_info_ph: dbn_output
        }

        # Set variables in the DQN to be those of Q and not Q^hat
        dqn.restore_q_hat_vars(dqn.variables_img_main, dqn.variables_img)

        # OPTIMIZE
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _ = dqn.sess.run([dqn.optimizer], feed_dict=vals, options=run_options,
                         run_metadata=run_metadata)

        # Store variables of Q in temporary data structure
        dqn.restore_q_hat_vars(dqn.variables_img, dqn.variables_img_main)

        # Print Metrics
        if iteration % metrics_freq == 0:
            # print timing information
            print(iteration, "time:", str(datetime.now()))

        if iteration % prediction_freq == 0:
            # evaluate system accuracy on train data set
            pred = dqn.sess.run(dqn.pred, feed_dict=vals)
            print("pred: ", pred)
            print("label: ", label_data)
            acc = dqn.sess.run(dqn.accuracy, feed_dict=vals)
            print("acc of train: ", acc)
            print()

        # Delayed System Updates
        if iteration % updates_freq == 0:
            # update variables in Q^hat to be the same as in Q
            dqn.assignVariables()

        if iteration % checkpoint_freq == 0:
            # save the model to checkpoint file
            dir_name = "omega_" + str(iteration / checkpoint_freq)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            dqn.saveModel(save_dir=dir_name)

    # FINISH
    # save final model to checkpoint file
    dir_name = "omega_final"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    dqn.saveModel(save_dir=dir_name)

    print("time end: ", datetime.now())
