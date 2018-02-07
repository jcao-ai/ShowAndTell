#coding:utf-8  
import tensorflow as tf  
from model import ShowAndTell, VizCallback
import keras
import os
import keras.backend as K

K.set_learning_phase(1)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("weight_path", "./keras_weight/weights_full.h5", "Weights data path")  
tf.app.flags.DEFINE_integer("initial_epoch", 0, "Start epoch at")  
tf.app.flags.DEFINE_integer("total_epoch", 100, "Total epochs to train")  
tf.app.flags.DEFINE_integer("steps_per_epoch", 18125, "Number of steps in each epoch")  
tf.app.flags.DEFINE_string("TFRecord_pattern", './tfrecords/train-?????-of-00256', "TFRecords file pattern")  

weight_path = FLAGS.weight_path
TFRecord_pattern = FLAGS.TFRecord_pattern

class VizCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(weight_path)
        print(epoch, 'Weights saved!')

def main(unused_argv):  
    initial_epoch = FLAGS.initial_epoch  
    total_epoch = FLAGS.total_epoch  
    steps_per_epoch = FLAGS.steps_per_epoch  
    model = ShowAndTell()
    model.config.input_file_pattern = TFRecord_pattern
    model.build_model()

    # Callback for saving weights on epoch end
    viz_cb = VizCallback()

    # Load weights if exists
    if os.path.isfile(weight_path):
        model.keras_model.load_weights(weight_path, by_name=True)

    model.keras_model.summary()
    model.keras_model.fit(epochs=total_epoch, steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch, y=np.zeros(model.config.batch_size), callbacks=[viz_cb])
  
if __name__ == '__main__':  
    tf.app.run()   # 解析命令行参数，调用main 函数 main(sys.argv)  
