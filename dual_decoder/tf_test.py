import os
import time
import itertools
import sys
import tensorflow as tf
import tf_model
import tf_hparams
import tf_metrics
import tf_input
from dual_encoder import dual_encoder_model

tf.flags.DEFINE_string("test_file", "./test.tfrecords", "Path of test data in TFRecords format")
tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("test_batch_size", 8, "Batch size for testing")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

tf.logging.set_verbosity(FLAGS.loglevel)

if __name__ == "__main__":
  hparams = tf_hparams.create_hparams()
  word_embeddings = tf_input.get_word_embeddings()
  model_fn = tf_model.create_model_fn(hparams,word_embeddings, model_impl=dual_encoder_model)
  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir,
    config=tf.contrib.learn.RunConfig())

  input_fn_test = tf_input.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[FLAGS.test_file],
    batch_size=FLAGS.test_batch_size,
    num_epochs=1)

  eval_metrics = tf_metrics.create_evaluation_metrics()
  estimator.evaluate(input_fn=input_fn_test, steps=None, metrics=eval_metrics)
