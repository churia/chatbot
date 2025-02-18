import os
import time
import itertools
import tensorflow as tf
import tf_model
import tf_hparams
import tf_input
import tf_metrics
from dual_encoder import dual_encoder_model

tf.flags.DEFINE_string("input_dir", "./", "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", None, "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 2000, "Evaluate after this many train steps")
FLAGS = tf.flags.FLAGS

TIMESTAMP = int(time.time())

if FLAGS.model_dir:
  MODEL_DIR = FLAGS.model_dir
else:
  MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "dev.tfrecords"))

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_argv):
  hparams = tf_hparams.create_hparams()

  model_fn = tf_model.create_model_fn(
    hparams,
    word_embedding=tf_input.get_word_embeddings(),
    model_impl=dual_encoder_model)

  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=MODEL_DIR,
    config=tf.contrib.learn.RunConfig())

  input_fn_train = tf_input.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    input_files=[TRAIN_FILE],
    batch_size=hparams.batch_size,
    num_epochs=FLAGS.num_epochs)
  
  input_fn_eval = tf_input.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[VALIDATION_FILE],
    batch_size=hparams.eval_batch_size,
    num_epochs=1)

  eval_metrics = tf_metrics.create_evaluation_metrics()

  eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    input_fn=input_fn_eval,
    eval_steps = 1,
    every_n_steps=FLAGS.eval_every,
    metrics=eval_metrics)
  
  estimator.fit(input_fn=input_fn_train, steps=10000)#, monitors=[eval_monitor])
  estimator.evaluate(input_fn=input_fn_eval, metrics=eval_metrics)
if __name__ == "__main__":
  tf.app.run()
