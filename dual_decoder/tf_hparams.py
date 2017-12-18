import tensorflow as tf
from collections import namedtuple


# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
#tf.flags.DEFINE_integer("rnn_dim", 100, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_content_len", 160, "Truncate contents to this length")
tf.flags.DEFINE_integer("max_response_len", 160, "Truncate response to this length")
#tf.flags.DEFINE_integer("max_response_len", 160, "Truncate response to this length")
tf.flags.DEFINE_string("cell_type", "RNN", "use either RNN or LSTM")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
#tf.flags.DEFINE_integer("batch_size", 128, "Batch size during training")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
#tf.flags.DEFINE_string("optimizer", "Adagrad", "Optimizer Name (Adam, Adagrad, etc)")
#tf.flags.DEFINE_string("optimizer", "SGD", "Optimizer Name (Adam, Adagrad, etc)")


FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_content_len",
    "max_response_len",
    "optimizer",
    "rnn_dim",
    "cell_type"
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
   # vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_content_len=FLAGS.max_content_len,
    max_response_len=FLAGS.max_response_len,
   # glove_path=FLAGS.glove_path,
   # vocab_path=FLAGS.vocab_path,
    rnn_dim=FLAGS.rnn_dim,
    cell_type=FLAGS.cell_type)
