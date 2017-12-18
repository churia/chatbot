import tensorflow as tf

TEXT_FEATURE_SIZE = 160

def get_feature_columns():
  feature_columns = []

  feature_columns.append(tf.contrib.layers.real_valued_column(
    column_name="content", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="content_len", dimension=1, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="response", dimension=TEXT_FEATURE_SIZE, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="response_len", dimension=1, dtype=tf.int64))

  feature_columns.append(tf.contrib.layers.real_valued_column(
    column_name="label", dimension=1, dtype=tf.int64))

  return set(feature_columns)

def input_fn(input_files, batch_size, num_epochs):
  features = tf.contrib.layers.create_feature_spec_for_parsing(
      get_feature_columns())

  feature_map = tf.contrib.learn.io.read_batch_features(
      file_pattern=input_files,
      batch_size=batch_size,
      features=features,
      reader=tf.TFRecordReader,
      randomize_input=True,
      num_epochs=num_epochs,
      queue_capacity=200000 + batch_size * 10,
      name="read_batch_features")

  print feature_map
  return feature_map

input_fn("train.tfrecords",64,1)
