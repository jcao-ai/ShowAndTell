#coding:utf-8  
import tensorflow as tf  
from model import ShowAndTell
import keras
import os
import keras.backend as K
import numpy as np

K.set_learning_phase(0)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("weight_path", "./keras_weight/weights_full.h5", "Weights data path")  
tf.app.flags.DEFINE_string("image_path", None, "Test image path to be tested with")  
tf.app.flags.DEFINE_integer("max_sentence_length", 20, "Max length of sentence to be predicted")  

weight_path = FLAGS.weight_path
image_path = FLAGS.image_path
max_length_of_sentence = FLAGS.max_sentence_length

def main(unused_argv):  
    if not os.path.isfile(image_path):
        print('Image file not found at: '+image_path)
        return

    model = ShowAndTell()
    model.config.mode = 'inference'
    model.build_model()

    # Load weights if exists
    if os.path.isfile(weight_path):
        model.inception_part_model.load_weights(weight_path, by_name=True)
        model.lstm_part_model.load_weights(weight_path, by_name=True)
    else:
        print('Weight file not found')
        return

    sess = K.get_session()
    with tf.gfile.FastGFile(image_path, "rb") as f:
        origin_img_val = model.process_image(f.read())
        test_image = tf.expand_dims(origin_img_val, 0)

    next_h_state = K.eval(model.inception_part_model([test_image])[0])
    next_c_state = K.eval(model.inception_part_model([test_image])[1])

    words = [model.vocab.start_id]
    for i in range(max_length_of_sentence):
        X, next_h_state, next_c_state = sess.run(model.lstm_part_model.outputs, feed_dict={"input_1:0": [[words[-1]]], "lstm_h_state:0": next_h_state, "lstm_c_state:0": next_c_state})
        words.append(np.argmax(X))
        if words[-1] == model.vocab.end_id:
            break
    print([model.vocab.id_to_word(word) for i, word in enumerate(words)])

if __name__ == '__main__':  
    tf.app.run()   # 解析命令行参数，调用main 函数 main(sys.argv)  
