# might need to put variables into the model somehow
# it would be nice if I can train and execute on a
# model in a separate folder

from __future__ import print_function

import tensorflow as tf 
from tensorflow.python.client import timeline
import numpy as np

from dqn_model_omega_np import *

from basic_tfrecord_rw import parse_sequence_example
from input_pipeline import *

import sys
import os
from os.path import isfile, join
from datetime import datetime
import math, random

import sys

GAMMA = 0.9
ALPHA = 1e-5
NUM_ITER = 30000
FOLDS = 1
NUM_REMOVED = 1

TEST_ITER = 50

if __name__ == '__main__':

	ts = datetime.now()
	print("time start: ", ts)
	#################################
	# Input params
	#################################
	graphbuild = [0]*TOTAL_PARAMS
	if(len(sys.argv) > 1):
		graphbuild[int(sys.argv[1])] = 1
	else:
		graphbuild = [1]*TOTAL_PARAMS

	#graphbuild = [0,0,1]
	if(sum(graphbuild) < 3):
		print("#########################")
		print("BUILDING PARTIAL MODEL")
		print("#########################")

	num_params = np.sum(graphbuild)

	#################################
	# Read contents of TFRecord file
	#################################
	
	path = "../tfrecords_balanced/"
	
	# all files (216)
	filenames = [f for f in os.listdir(path) if isfile(join(path, f))]
	#filenames = ["prompt.tfrecord", "reward.tfrecord", "abort.tfrecord"]
	filenames = [path +x for x in filenames ]
	filenames.sort()
		
	#################################
	# Generate Model
	#################################
	dqn_chkpnt = "./omega_2/model.ckpt"#./beta_3/model.ckpt"
	dqn = DQNModel(graphbuild, batch_size=BATCH_SIZE, learning_rate=ALPHA, filename=dqn_chkpnt, log_dir="LOG_DIR")

	if(dqn_chkpnt != ""):
		dqn.assignVariables()

	#################################
	# Train Model
	#################################
	
	coord = tf.train.Coordinator()

	#sequence length - slen
	#sequence length prime- slen_pr
	#image raw - i
	#points raw - p
	#audio raw - a
	#previous action - pl
	#action - l
	#image raw prime - i_pr
	#points raw prime - p_pr
	#audio raw prime - a_pr
	
	slen, slen_pr, i, p, a, pl, l, i_pr, p_pr, a_pr, n_id = input_pipeline(filenames)
	l = tf.squeeze(l, [1])
	
	dqn.sess.run(tf.local_variables_initializer())#initializes batch queue
	dqn.sess.graph.finalize()
	threads = tf.train.start_queue_runners(coord=coord, sess=dqn.sess)

	print("Num epochs: "+str(NUM_EPOCHS)+", Batch Size: "+str(BATCH_SIZE)+", Num Files: "+str(len(filenames))+", Num steps: "+str(NUM_ITER))
	for iteration in range(NUM_ITER):
		ts_it = datetime.now()
		n_seq, n_seq2, img_data, pnt_data, aud_data, num_prompts, label_data, img_data2, pnt_data2, aud_data2, name \
				= dqn.sess.run([slen, slen_pr, i, p, a, pl, l, i_pr, p_pr, a_pr, n_id])
		#print(name)
		#generate partition information in order to get the prediction from LSTM
		
		partitions_1 = np.zeros((BATCH_SIZE, np.max(n_seq)))
		partitions_2 = np.zeros((BATCH_SIZE, np.max(n_seq2)))

		for x in range(BATCH_SIZE):
			
			if(np.max(n_seq) > 0):
				v = n_seq[x]-1
				if v < 0:
					v = 0
				partitions_1[x][v] = 1
			
			if(np.max(n_seq2) > 0):
				v = n_seq2[x]-1
				if v < 0:
					v = 0
				partitions_2[x][v] = 1
		
		#modify rewards for non-terminal states
		newy = 0
		if(np.max(n_seq2) > 1):
			#print("set_shape")
			img_data2 = set_shape(img_data2, img_dtype)
			pnt_data2 = set_shape(pnt_data2, pnt_dtype)
			aud_data2 = set_shape(aud_data2, aud_dtype)
			#print("set_shape2")
			vals = {
				dqn.seq_length_ph: n_seq2, 
				dqn.img_ph: img_data2, 
				dqn.pnt_ph: pnt_data2, 
				dqn.aud_ph: aud_data2
				,dqn.partitions_ph: partitions_2
				,dqn.train_ph: False
				,dqn.prompts_ph: np.sign(n_seq2) }
			#print(img_data.shape, pnt_data.shape, aud_data.shape, img_data2.shape, pnt_data2.shape, aud_data2.shape)
			newy = dqn.sess.run(dqn.max_q_hat, feed_dict=vals)#get_max_q
			newy *= np.sign(n_seq2)
			
		else:
			newy = np.zeros(BATCH_SIZE)
		
		r = np.array(label_data) # an array equal in length to batch that describes the reward being received in the next state
				
		r[:,0] = r[:,0]*.2
	
		for j in range(r.shape[0]):
			for v in range(r.shape[1]):
				if r[j][v] != 0:
					if(v < 2):
						r[j][v]+= newy[j] * GAMMA
		
		vals = {
			dqn.seq_length_ph: n_seq, 
			dqn.img_ph: img_data,
			dqn.pnt_ph: pnt_data, 
			dqn.aud_ph: aud_data, 
			dqn.y_ph: r,
			dqn.partitions_ph: partitions_1,
			dqn.train_ph: True,
			dqn.prompts_ph: num_prompts
			}

		prep_t = datetime.now() - ts_it
		switch_s = datetime.now()
		#restore main to img
		dqn.restore_q_hat_vars(dqn.variables_img_main, dqn.variables_img)
		switch_t = datetime.now() - switch_s
		# OPTIMIZE
		option_t = datetime.now()
		run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
		run_metadata=tf.RunMetadata()
		option_t = datetime.now() - option_t
		opt_s = datetime.now()
		#summary, _ = dqn.sess.run([dqn.merged_summary, dqn.optimizer], feed_dict=vals, options=run_options, run_metadata=run_metadata)
		_ = dqn.sess.run([dqn.optimizer], feed_dict=vals, options=run_options, run_metadata=run_metadata)
		
		opt_t = datetime.now() - opt_s
		#dqn.train_writer.add_run_metadata(run_metadata, 'step%03d' % iteration)
		

		#restore img to main

		dqn.restore_q_hat_vars(dqn.variables_img, dqn.variables_img_main)
		'''
		ce_s = datetime.now()
		vals[dqn.train_ph] = False
		ce = dqn.sess.run(dqn.cross_entropy, feed_dict=vals)
		ce_t = datetime.now() - ce_s
		'''
		if(iteration%1 == 0):
			print(iteration, "total_time:", datetime.now()-ts_it, "prep_time:",prep_t, "switch_time:",switch_t, "optimization_time:",opt_t, "option_time:",option_t)
			#print("label:\n", label_data)
			#print(name)
		if(iteration%100 == 0):
			pred = dqn.sess.run(dqn.pred, feed_dict=vals)
			print("pred: ", pred)
			print("label: ", label_data)
			print("--------")

			acc = dqn.sess.run(dqn.accuracy, feed_dict=vals)
			print("acc of train: ", acc)
		
		if(iteration % 100 == 0):
			dqn.assignVariables()
			
		
			if(iteration % 1000 == 0):
				#overwrite the saved model until 10,000 iterations have passed
				dir_name = "omega_"+str((iteration / 10000)+1)
				if not os.path.exists(dir_name):
					os.makedirs(dir_name)
				dqn.saveModel(save_dir=dir_name)
			
	#######################
	## FINALIZE TRAINING
	#######################
	
	dir_name = "omega_final"
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)
	dqn.saveModel(save_dir=dir_name)

	te = datetime.now()
	print("time end: ", te)
	print("elapsed: ", te-ts)
	
