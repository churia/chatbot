import tensorflow as tf
import sys

def get_id_feature(features, key, len_key,max_len):
  ids = features[key]
  ids_len = tf.squeeze(features[len_key], [1])
  ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
  return ids, ids_len

def create_train_op(loss, hparams):
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.train.get_global_step(),
      learning_rate=hparams.learning_rate,
      clip_gradients=10.0,
      optimizer=hparams.optimizer)
  return train_op


def create_model_fn(hparams, word_embedding, model_impl):

  def model_fn(features, targets, mode):
    content, content_len = get_id_feature(
        features, "content", "content_len", hparams.max_content_len)

    response, response_len = get_id_feature(
        features, "response", "response_len", hparams.max_content_len)

    batch_size = targets.get_shape().as_list()[0]

    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      probs, loss = model_impl(
          hparams,
          mode,
          word_embedding,
          content,
          content_len,
          response,
          response_len,
          targets)
      train_op = create_train_op(loss, hparams)
      return probs, loss, train_op

    if mode == tf.contrib.learn.ModeKeys.INFER:
      probs, loss = model_impl(
          hparams,
          mode,
          word_embedding,
          content,
          content_len,
          response,
          response_len,
          None)
      return probs, 0.0, None

    if mode == tf.contrib.learn.ModeKeys.EVAL:

      # We have 10 exampels per record, so we accumulate them
      all_contents = [content]
      all_content_lens = [content_len]
      all_responses = [response]
      all_response_lens = [response_len]
      all_targets = [tf.ones([batch_size, 1], dtype=tf.int64)]

      for i in range(9):
        distractor, distractor_len = get_id_feature(features,
            "distractor_{}".format(i),
            "distractor_{}_len".format(i),
            hparams.max_response_len)
        all_contents.append(content)
        all_content_lens.append(content_len)
        all_responses.append(distractor)
        all_response_lens.append(distractor_len)
        all_targets.append(
          tf.zeros([batch_size, 1], dtype=tf.int64)
        )

      probs, loss = model_impl(
          hparams,
          mode,
          word_embedding,
          tf.concat(all_contents,0),
          tf.concat(all_content_lens,0),
          tf.concat(all_responses,0),
          tf.concat(all_response_lens,0),
          tf.concat(all_targets,0))

      split_probs = tf.split(probs, 10, 0)
      shaped_probs = tf.concat(split_probs,1)

      # Add summaries
      tf.summary.histogram("eval_correct_probs_hist", split_probs[0])
      tf.summary.scalar("eval_correct_probs_average", tf.reduce_mean(split_probs[0]))
      tf.summary.histogram("eval_incorrect_probs_hist", split_probs[1])
      tf.summary.scalar("eval_incorrect_probs_average", tf.reduce_mean(split_probs[1]))

      return shaped_probs, loss, None

  return model_fn
