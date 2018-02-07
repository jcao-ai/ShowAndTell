#!/usr/bin/python
#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import coremltools
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Reshape, Lambda, Embedding, LSTM
from keras import backend as K
from model import ModelConfiguration, Vocabulary
from inception_v3 import InceptionV3

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("weight_path", "./keras_weight/weights_full.h5", "Weights data path")  
tf.app.flags.DEFINE_bool("export_lstm", True, "Whether export lstm part or inception part model")  
weight_path = FLAGS.weight_path
export_lstm = FLAGS.export_lstm

K.set_learning_phase(0)


# Here we define model again because coremltools isn't friendly enough to reuse the model. 
# Error is raised when reusing the model architecture in model.py
# So here is the tricky/magic part.
class Model2CoreML(object):
    def __init__(self):
        self.vocab = Vocabulary('./word_counts.txt')
        self.config = ModelConfiguration()

    def build_input(self):
        input_seqs = Input(shape=(1,), dtype='int32')
        images = Input(shape=(self.config.image_height, self.config.image_width, 3), dtype='float32')
        
        self.images = images
        self.input_seqs = input_seqs
        
    # Inception model part encode the input image into tensor in shape of (batch, 1, 512), and give it to LSTM cell to get the `state`
    # Image -> {Inception_V3} -> (batch, 1, 512) -> {LSTM} -> Output: lstm_output, h_state, c_state
    def build_inception_part_model(self):
        self.build_input()

        image_input = Input(tensor=self.images)
        seq_input = Input(tensor=self.input_seqs)

        raw_vision_model = InceptionV3(include_top=True, input_tensor=image_input)

        image_embedding = Dense(self.config.embedding_size, activation=None, kernel_initializer='random_uniform',
                bias_initializer='zeros')(raw_vision_model.outputs[0])
        init_lstm_input = Reshape(target_shape=(1, self.config.embedding_size), name='reshape')(image_embedding)

        # LSTM Cell
        lstm_cell = LSTM(self.config.num_lstm_units, unroll=False, return_sequences=True, dropout=0.4, recurrent_dropout=0.4, return_state=True)
        
        # Get cell state from encoded image input
        initial_state_from_image = lstm_cell(init_lstm_input)
        lstm_output, h_state, c_state = initial_state_from_image

        # BUG coremltools only accept `lstm_output` but not h_state/c_state as model's output
        # Although in Xcode the model's output will still be (output, h_state, c_state)
        inception_part_model = Model(inputs=[image_input], outputs=[lstm_output])
        inception_part_model.load_weights('./keras_weight/weights_full.h5', by_name=True)
        self.inception_part_model = inception_part_model

    #
    # This part model is responsible for predict the next word with last LSTM state and last most possible word.
    #
    # Whole unroll progress:
    #    LSTM(encoded_image, zero_state) -> _ , h_state_1, c_state_1
    #    LSTM([<s>], h_state_1, c_state_1) -> [a], h_state_2, c_state_2
    #    LSTM([a], h_state_2, c_state_2) -> [dog], h_state_3, c_state_3
    #    LSTM([dog], h_state_3, c_state_3) -> [is], h_state_4, c_state_4
    #    LSTM([is], h_state_4, c_state_4) -> [eating], h_state_4, c_state_4
    #    ...
    #    LSTM([of], h_state_m, c_state_m) -> [pizza], h_state_n, c_state_n
    #    LSTM([pizza], h_state_n, c_state_n) -> [</s>], h_state_n, c_state_n
    #
    # `<s>` is start word and `</s>` is end word of the whole sentence
    #
    # Comlpete Sentence: <s> A dog is eating a piece of pizza </s>
    def build_lstm_part_model(self):
        self.build_input()

        image_input = Input(tensor=self.images)
        seq_input = Input(tensor=self.input_seqs)

        step_h_state = Input(batch_shape=(None, 512,))
        step_c_state = Input(batch_shape=(None, 512,))

        # Embed the vocabulary dimension to 512 dimensions
        seq_embeddings = Embedding(input_dim=self.config.vocab_size, output_dim=self.config.embedding_size, mask_zero=True)(seq_input)

        raw_vision_model = InceptionV3(include_top=True, input_tensor=image_input)
        image_embedding = Dense(self.config.embedding_size, activation=None, kernel_initializer='random_uniform',
                bias_initializer='zeros')(raw_vision_model.outputs[0])

        lstm_cell = LSTM(self.config.num_lstm_units, unroll=False, return_sequences=True, dropout=0.4, recurrent_dropout=0.4, return_state=True)
        

        training_output, next_h_state, next_c_state = lstm_cell(seq_embeddings)
        
        ### NOTICE: 
        # Softmax activation is included here but not in model.py
        # because logits is needed in loss function.
        masked_training_output = Dense(self.config.vocab_size, activation='softmax', kernel_initializer='random_uniform',
               bias_initializer='zeros')(training_output)

        lstm_part_model = Model(inputs=[seq_input], outputs=[masked_training_output])
        lstm_part_model.load_weights('./keras_weight/weights_full.h5', by_name=True)
        self.lstm_part_model = lstm_part_model
        
def convert_lstm_part_model(wrapper):
    print('converting...')
    wrapper.build_lstm_part_model()
    model = wrapper.lstm_part_model
    coreml_model = coremltools.converters.keras.convert(model)
    coreml_model.author = 'Tsao'
    coreml_model.license = 'MIT'
    coreml_model.short_description = 'Show and tell.'
    
    coreml_model.save('lstm_part_model.mlmodel')
    print('lstm_part_model converted.')

def convert_inception_part_model(wrapper):
    print('converting...')
    wrapper.build_inception_part_model()
    model = wrapper.inception_part_model

    coreml_model = coremltools.converters.keras.convert(model, input_names=['image'], image_input_names='image', image_scale=1/128.0, red_bias=-1, blue_bias=-1, green_bias=-1)
    coreml_model.author = 'Tsao'
    coreml_model.license = 'MIT'
    coreml_model.short_description = 'Show and tell.'

    coreml_model.save('inception_part_model.mlmodel')
    print('inception_part_model converted.')



def main(unused_argv):  
    wrapper = Model2CoreML()
    wrapper.config.mode = 'inference'
    if export_lstm:
        convert_lstm_part_model(wrapper)
    else:
        convert_inception_part_model(wrapper)

if __name__ == '__main__':  
    tf.app.run()   # 解析命令行参数，调用main 函数 main(sys.argv)  