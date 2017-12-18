import tensorflow as tf
import numpy as np
import sys
import os

reload(sys)
sys.setdefaultencoding('Cp1252')

def dual_encoder_model(hparams, mode, embeddings, content, content_len, response, response_len, label):

	word_embeddings = tf.get_variable("word_embeddings",initializer=embeddings)
	#embed the content and the response
	content_embeded = tf.nn.embedding_lookup(word_embeddings,content,name="content_embeded")
	response_embeded = tf.nn.embedding_lookup(word_embeddings,response,name="response_embeded")


	with tf.variable_scope("rnn") as vs:
		if hparams.cell_type == "LSTM":
			#cell = tf.nn.rnn_cell.LSTMCell(hparams.rnn_dim)
			cell = tf.nn.rnn_cell.LSTMCell(hparams.rnn_dim,forget_bias=2.0,use_peepholes=True,state_is_tuple=True)
			#cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)
			
			# Run the utterance and context through the RNN
			rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
				cell, 
				tf.concat([content_embeded, response_embeded],0),
				sequence_length=tf.concat([content_len, response_len],0),
				dtype=tf.float32)
			
			encoding_content, encoding_response = tf.split(rnn_states.h, 2, 0)
		else:
			cell =  tf.nn.rnn_cell.BasicRNNCell(hparams.rnn_dim)
			content_outputs, encoding_content = tf.nn.dynamic_rnn(
				cell, 
				content_embeded,
				sequence_length=content_len,
				dtype=tf.float32)

			response_outputs, encoding_response = tf.nn.dynamic_rnn(
				cell, 
				response_embeded,
				sequence_length=response_len,
				dtype=tf.float32)
			

	with tf.variable_scope("prediction") as vs:
		M = tf.get_variable("M",shape=[hparams.rnn_dim,hparams.rnn_dim],initializer=tf.truncated_normal_initializer())
		
		# "Predict" a response: c * M
		generated_response = tf.matmul(encoding_content, M)
		generated_response = tf.expand_dims(generated_response, 2)
		encoding_response = tf.expand_dims(encoding_response, 2)
		
		# Dot product between generated response and actual response/candidates
		logits = tf.matmul(generated_response, encoding_response, True)
		logits = tf.squeeze(logits,[2])
		
		#Apply sigmoid to convert score to probabilities
		probs = tf.sigmoid(logits)
		
		if mode == tf.contrib.learn.ModeKeys.INFER:
			return probs, None
		# Calculate the binary cross-entropy loss
		losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(label))	
		
	# Mean loss across the batch of examples
	mean_loss = tf.reduce_mean(losses, name="mean_loss")
	
	return probs, mean_loss		

