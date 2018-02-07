#!/usr/bin/python
#coding:utf-8
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import LSTM
from keras.layers import Reshape, Lambda, Embedding
from keras.layers.merge import add, concatenate
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
# from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import numpy as np
from inception_v3 import InceptionV3
import image_processing
import inputs as input_ops
import os
from keras.utils import plot_model
import keras

OUTPUT_DIR = 'keras_weights'
K.set_learning_phase(0)

class Vocabulary(object):
    """Vocabulary class for an image-to-text model."""
    def __init__(self,
                vocab_file,
                start_word="<S>",
                end_word="</S>",
                unk_word="<UNK>"):
        """Initializes the vocabulary.

        Args:
        vocab_file: File containing the vocabulary, where the words are the first
            whitespace-separated token on each line (other tokens are ignored) and
            the word ids are the corresponding line numbers.
        start_word: Special word denoting sentence start.
        end_word: Special word denoting sentence end.
        unk_word: Special word denoting unknown words.
        """
        if not tf.gfile.Exists(vocab_file):
            tf.logging.fatal("Vocab file %s not found.", vocab_file)
        tf.logging.info("Initializing vocabulary from file: %s", vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = list(f.readlines())
        reverse_vocab = [line.split()[0] for line in reverse_vocab]
        if unk_word not in reverse_vocab:
            reverse_vocab.append(unk_word)
        vocab = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])

        print("Created vocabulary with %d words" % len(vocab))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

        # Save special word ids.
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = vocab[unk_word]
        self.start_word = start_word
        self.end_word = end_word

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self.reverse_vocab):
            return self.reverse_vocab[self.unk_id]
        else:
            return self.reverse_vocab[word_id]

    def size(self):
        return len(self.vocab)

class ModelConfiguration(object):
  def is_training(self):
    return self.mode == 'train'
  def __init__(self):
    self.mode = "train"
    self.embedding_size = 512
    self.num_lstm_units = 512
    self.lstm_dropout_keep_prob = 0.6
    self.initializer = tf.random_uniform_initializer(
        minval=-0.08,
        maxval=0.08)
    self.input_file_pattern = './tfrecords/train-?????-of-00256'
    self.batch_size = 32
    self.image_format = "jpeg"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard = 2300
    # Minimum number of shards to keep in the input queue.
    self.input_queue_capacity_factor = 2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads = 1

    # Name of the SequenceExample context feature containing image data.
    self.image_feature_name = "image/data"
    # Name of the SequenceExample feature list containing integer captions.
    self.caption_feature_name = "image/caption_ids"

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.vocab_size = 12000

    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4
    self.initial_learning_rate = 1e-4

    self.image_height = 299
    self.image_width = 299

    self.max_word_length = 40

def softmax_sparse_loss(args):
    output, target, mask = args
    cross_entropy = K.sparse_categorical_crossentropy(output=output, target=target, from_logits=True)
    return K.sum(cross_entropy * K.cast(mask, dtype='float32'), keepdims=False) / K.sum(K.cast(mask, dtype='float32'),keepdims=False)

class VizCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights('./keras_weight/weights_full.h5')
        print(epoch, ' on_epoch_end: save done!')


class ShowAndTell(object):
    def __init__(self):
        self.vocab = Vocabulary('./word_counts.txt')
        self.config = ModelConfiguration()
        self.reader = tf.TFRecordReader()

    def process_image(self, encoded_image, thread_id=0):
        """Decodes and processes an image string.

        Args:
        encoded_image: A scalar string Tensor; the encoded image.
        thread_id: Preprocessing thread id used to select the ordering of color
            distortions.

        Returns:
        A float32 Tensor of shape [height, width, 3]; the processed image.
        """
        return image_processing.process_image(encoded_image,
                                            is_training=self.config.is_training(),
                                            height=self.config.image_height,
                                            width=self.config.image_width,
                                            thread_id=thread_id,
                                            image_format=self.config.image_format)

    def build_input(self):
        if self.config.mode == "inference":
            input_seqs = Input(shape=(1,), dtype='int32')
            images = Input(shape=(self.config.image_height, self.config.image_width,3,), dtype='float32')
        else:
            # Prefetch serialized SequenceExample protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.config.input_file_pattern,
                is_training=self.config.is_training(),
                batch_size=self.config.batch_size,
                values_per_shard=self.config.values_per_input_shard,
                input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                num_reader_threads=self.config.num_input_reader_threads)

            # Image processing and random distortion. Split across multiple threads
            # with each thread applying a slightly different distortion.
            assert self.config.num_preprocess_threads % 2 == 0
            images_and_captions = []
            for thread_id in range(self.config.num_preprocess_threads):
                serialized_sequence_example = input_queue.dequeue()
                encoded_image, caption = input_ops.parse_sequence_example(
                    serialized_sequence_example,
                    image_feature=self.config.image_feature_name,
                    caption_feature=self.config.caption_feature_name)
                image = self.process_image(encoded_image, thread_id=thread_id)
                images_and_captions.append([image, caption])

            # Batch inputs.
            queue_capacity = (2 * self.config.num_preprocess_threads *
                                self.config.batch_size)
            images, input_seqs, target_seqs, input_mask = (
                input_ops.batch_with_dynamic_pad(images_and_captions,
                                                batch_size=self.config.batch_size,
                                                queue_capacity=queue_capacity))
            self.target_seqs = target_seqs
            self.input_mask = input_mask
        self.images = images
        self.input_seqs = input_seqs
        
    def build_model(self):
        self.build_input()

        sess = K.get_session()
        tf.train.start_queue_runners(sess)

        image_input = Input(tensor=self.images)
        seq_input = Input(tensor=self.input_seqs)
        if self.config.is_training():
            seq_targets = Input(tensor=self.target_seqs)
            input_masks = Input(tensor=self.input_mask)
                    
        # Embed the vocabulary dimension to 512 dimensions
        # seq_embeddings' shape: (batch_size, 1, 512)
        seq_embeddings = Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embedding_size, mask_zero=True)(seq_input)

        vision_model = InceptionV3(include_top=True, input_tensor=image_input)
        for layer in vision_model.layers:
            layer.trainable = False

        image_embedding = Dense(self.config.embedding_size, activation=None, kernel_initializer='random_uniform',
                bias_initializer='zeros')(vision_model.outputs[0])
        init_lstm_input = Reshape(target_shape=(1, self.config.embedding_size), name='reshape')(image_embedding)

        # TODO There is a bug when LSTM is initialized with `dropout`/'recurrent_dropout'
        # If error raised please check the following issue link:
        # `https://github.com/keras-team/keras/issues/8407#issuecomment-361901801`
        lstm_cell = LSTM(self.config.num_lstm_units, unroll=False, return_sequences=True, dropout=0.4, recurrent_dropout=0.4, return_state=True)
        
        initial_state_from_image = lstm_cell(init_lstm_input)
        lstm_output, h_state, c_state = initial_state_from_image

        if self.config.is_training():
            training_output = lstm_cell(seq_embeddings, initial_state=[h_state, c_state])[0]
        else:
            inception_part_model = Model(inputs=[image_input], outputs=[h_state, c_state])
            self.inception_part_model = inception_part_model

            step_h_state = Input(batch_shape=(None, 512,), name='lstm_h_state')
            step_c_state = Input(batch_shape=(None, 512,), name='lstm_c_state')

            training_output, next_h_state, next_c_state = lstm_cell(seq_embeddings, initial_state=[step_h_state, step_c_state])
 
        masked_training_output = Dense(self.config.vocab_size, activation=None, kernel_initializer='random_uniform',
            bias_initializer='zeros')(training_output)

        if self.config.is_training():
            loss_out = Lambda(softmax_sparse_loss, name='softmax_loss')([masked_training_output, seq_targets, input_masks])
            model = Model(inputs=[seq_input, seq_targets, input_masks, image_input], outputs=[loss_out])
            model.summary()
            model.compile(loss={'softmax_loss': lambda y_true, y_pred: y_pred}, optimizer='adam')
            self.keras_model = model
        else:
            lstm_part_model = Model(inputs=[seq_input, step_h_state, step_c_state], outputs=[masked_training_output, next_h_state, next_c_state])
            self.lstm_part_model = lstm_part_model
        