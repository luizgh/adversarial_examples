import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2
import inception_v4
import densenet
import numpy as np

class densenet_model:
    def __init__(self, model_path, model='denset169', new_scope=None):
        self.image_size = 299
        self.num_labels = 1000
        self.num_channels = 3
        self.new_scope = new_scope
        
        assert model in ['densenet169']
        if model == 'densenet169':
            self.arg_scope = densenet.densenet_arg_scope
            self.inception_model = densenet.densenet169
            self.scope='densenet169'


        x_input = tf.placeholder(tf.float32, shape=(None,self.image_size,self.image_size,self.num_channels))

        if self.new_scope is None:
           self.new_scope = self.scope 

        self.model_path = model_path
        self.mean = np.array([123.68, 116.78, 103.94]).reshape(1,1,1,3)
        self.scale_factor = 0.017
        self.first = True

    def initialize(self, sess):
        self.saver.restore(sess, self.model_path)

    def predict(self, img):
        # from inception preprocess to densenet preprocess:
        img = ((img+1) *127.5 - self.mean) * self.scale_factor
        reuse = not self.first
        with slim.arg_scope(self.arg_scope()):
          _, end_points = self.inception_model(
              img, num_classes=self.num_labels, is_training=False, reuse=reuse, scope=self.new_scope)

        if self.first:
          if self.scope != self.new_scope:
            var_dict = {var.op.name.replace(self.new_scope, self.scope, 1): var 
                for var in slim.get_model_variables(scope=self.new_scope)}
          else:
            var_dict = slim.get_model_variables(scope=self.scope)

          self.saver = tf.train.Saver(var_dict)
        self.first = False

        return end_points['Logits']

class inception_model:
    def __init__(self, model_path, model='inception_v3', new_scope=None):
        self.image_size = 299
        self.num_labels = 1001
        self.num_channels = 3
        self.new_scope = new_scope
        
        assert model in ['inception_v3', 'inception_v4', 'inception_resnet_v2', 'densenet']
        if model == 'inception_v3':
            self.arg_scope = inception.inception_v3_arg_scope
            self.inception_model = inception.inception_v3
            self.scope='InceptionV3'
        elif model == 'inception_v4':
            self.arg_scope = inception_v4.inception_v4_arg_scope
            self.inception_model = inception_v4.inception_v4
            self.scope='InceptionV4'
        elif model == 'inception_resnet_v2':
            self.arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope
            self.inception_model = inception_resnet_v2.inception_resnet_v2
            self.scope='InceptionResnetV2'
        elif model == 'densenet':
            self.arg_scope = densenet.densenet_arg_scope
            self.inception_model = densenet.densenet169
            self.scope='densenet169'


        x_input = tf.placeholder(tf.float32, shape=(None,self.image_size,self.image_size,self.num_channels))

        if self.new_scope is None:
           self.new_scope = self.scope 

        self.first = True
        self.model_path = model_path

    def initialize(self, sess):
        self.saver.restore(sess, self.model_path)

    def predict(self, img):
        reuse = not self.first
        with slim.arg_scope(self.arg_scope()):
          _, end_points = self.inception_model(
              img, num_classes=self.num_labels, is_training=False, reuse=reuse, scope=self.new_scope)

        if self.first:
          if self.scope != self.new_scope:
            var_dict = {var.op.name.replace(self.new_scope, self.scope, 1): var 
                for var in slim.get_model_variables(scope=self.new_scope)}
          else:
            var_dict = slim.get_model_variables(scope=self.scope)

          self.saver = tf.train.Saver(var_dict)
        self.first = False

        return end_points['Logits']
