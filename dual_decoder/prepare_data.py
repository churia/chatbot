import os,sys
import tensorflow as tf
import numpy as np

MAX_SENTENCE_LEN = 160
SPLIT = "+++$+++"

FASTTEXT_PATH = sys.argv[1]
TRAIN_PATH = sys.argv[2]
EVAL_PATH = sys.argv[4]
TEST_PATH = sys.argv[5]

def transform_content(sentence, vocab):
  '''
  map content sentence to fixed length sequence of word id, 
  use the last part of response if the sentence is too long
  '''
  seq=[vocab[w] for w in sentence.split()]
  length = len(seq)

  if length > MAX_SENTENCE_LEN:
    seq = seq[length-MAX_SENTENCE_LEN:]
  
  if length < MAX_SENTENCE_LEN:
    seq += [0]*(MAX_SENTENCE_LEN-length)

  length = min(length,MAX_SENTENCE_LEN)
  if (len(seq)!=160):
    print "error"
  return length, seq


def transform_response(sentence, vocab):
  '''
  map response sentence to to fixed length sequence of word id,
  use the first part of response if the sentence is too long
  '''
  seq=[vocab[w] for w in sentence.split()]
  length = len(seq)

  if length> MAX_SENTENCE_LEN:
    seq = seq[:MAX_SENTENCE_LEN]
 
  if length < MAX_SENTENCE_LEN:
    seq += [0]*(MAX_SENTENCE_LEN-length)

  length = min(length,MAX_SENTENCE_LEN)
  if (len(seq)!=160):
    print "error"
  return length, seq


def create_example_train(line, vocab):
  """
  Creates a training example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  [content, response, label] = line.strip().split(SPLIT)
  content_len, content_transformed = transform_content(content, vocab)
  response_len, response_transformed = transform_response(response, vocab)

  # New Example
  example = tf.train.Example()
  example.features.feature["content"].int64_list.value.extend(content_transformed)
  example.features.feature["response"].int64_list.value.extend(response_transformed)
  example.features.feature["content_len"].int64_list.value.extend([content_len])
  example.features.feature["response_len"].int64_list.value.extend([response_len])
  example.features.feature["label"].int64_list.value.extend([int(label)])
  return example


def create_example_test(line, vocab):
  """
  Creates a test/validation example for the Ubuntu Dialog Corpus dataset.
  Returnsthe a tensorflow.Example Protocol Buffer object.
  """
  strs = line.strip().split(SPLIT)
  content, response = strs[:2]
  distractors = strs[2:]
  
  content_len, content_transformed = transform_content(content, vocab)
  response_len, response_transformed = transform_response(response, vocab)

  # New Example
  example = tf.train.Example()
  example.features.feature["content"].int64_list.value.extend(content_transformed)
  example.features.feature["response"].int64_list.value.extend(response_transformed)
  example.features.feature["content_len"].int64_list.value.extend([content_len])
  example.features.feature["response_len"].int64_list.value.extend([response_len])

  # Distractor sequences
  for i, distractor in enumerate(distractors):
    dis_key = "distractor_{}".format(i)
    dis_len_key = "distractor_{}_len".format(i)
    dis_len, dis_transformed = transform_response(distractor, vocab)
    example.features.feature[dis_len_key].int64_list.value.extend([dis_len])
    example.features.feature[dis_key].int64_list.value.extend(dis_transformed)
  return example


def create_tfrecords_file(mode, input_filename, output_filename, vocab):
  """
  Creates a TFRecords file for the given input data and
  example transofmration function
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  print("Creating TFRecords file at {}...".format(output_filename))

  if mode == "test":
    with open(input_filename) as f:
      f.readline()
      for l in f.readlines():
        x = create_example_test(l,vocab)
        writer.write(x.SerializeToString())  
  else:
    with open(input_filename) as f:
      f.readline()
      for l in f.readlines():
        x = create_example_train(l,vocab)
        writer.write(x.SerializeToString())

  writer.close()
  print("Wrote to {}".format(output_filename))

def run_fasttext(fasttext_path,hparam="-minCount 1"):
  os.system(fasttext_path+"/fasttext skipgram -input fast.input -output word_embedding "+hparam)

def word_embedding(fasttext_path,trainfile, testfiles):
  #prepare doc for fasttext input using train/test files
  if not os.path.exists('fast.input'):
    doc = []
    with open(trainfile) as f:
      f.readline()
      for l in f.readlines():
        doc.append(" ".join(l.rstrip().split(SPLIT)[:-1]))
    for fil in testfiles:
      with open(fil) as f:
        f.readline()
        for l in f.readlines():
          doc.append(" ".join(l.rstrip().split(SPLIT)))

    with open('fast.input','w') as f:
      for line in doc:
        f.write(line+"\n")

  #run fasttext to get word embeddings
  run_fasttext(fasttext_path)

def get_vocab(fasttext_path,trainfile, testfiles):
  #use fasttext to get word embeddings based on train&test files
  if not os.path.exists("word_embedding.vec"):
    word_embedding(fasttext_path,trainfile, testfiles)

  vocab = {}
  with open("word_embedding.vec") as f:
    for i, line in enumerate(f.readlines()):
      vocab[line.split()[0]] = i

  return vocab

if __name__ == "__main__":


  print("Get vocabulary from word_embedding...")
  vocab = get_vocab(FASTTEXT_PATH,TRAIN_PATH,[TEST_PATH,EVAL_PATH])


  # Create test.tfrecords
#  create_tfrecords_file("test",TEST_PATH,"test.tfrecords",vocab)

  # Create dev.tfrecords
#  create_tfrecords_file("test",EVAL_PATH,"dev.tfrecords",vocab)

  # Create train.tfrecords
  create_tfrecords_file("train",TRAIN_PATH,"train.tfrecords",vocab)
