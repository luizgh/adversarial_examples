import tensorflow as tf
import numpy as np

class PGD_attack:
    def __init__(self, model, batchsize, max_epsilon, maxiter, targeted=False, initial_lr=0.5, lr_decay=0.98):
        self.x_input = tf.placeholder(tf.float32, shape=(None,299,299,3))
        self.y_input = tf.placeholder(tf.int32, shape=(None))
        
        self.lower_bounds = tf.placeholder(tf.float32, shape=(None,299,299,3))
        self.upper_bounds = tf.placeholder(tf.float32, shape=(None,299,299,3))

        self.delta = tf.Variable(np.zeros((batchsize, 299, 299, 3), dtype=np.float32), name='delta')

        self.adv_img = self.x_input + self.delta

        self.probs = model.predict(self.adv_img)

        row_indices = tf.range(batchsize)

        indices = tf.transpose([row_indices, self.y_input])
        self.pred = tf.gather_nd(self.probs, indices)
        self.grad = tf.gradients(tf.reduce_sum(self.pred), self.x_input)

        self.lr = tf.Variable(initial_lr)

        if targeted:
            multiplier =  1.
        else:
            multiplier = -1.

        delta_new = self.delta + (multiplier) * self.lr * self.grad[0]
        delta_new = tf.maximum(delta_new, self.lower_bounds)
        self.delta_new = tf.minimum(delta_new, self.upper_bounds)
        self.train_op = tf.assign(self.delta, self.delta_new)
        self.update_lr = tf.assign(self.lr, self.lr * lr_decay)
        
        self.maxiter = maxiter
        self.max_epsilon = max_epsilon
        self.batchsize = batchsize
        self.initial_lr = initial_lr

        self.delta_to_assign = tf.placeholder(tf.float32)
        self.assign_delta = self.delta.assign(self.delta_to_assign)
        self.reset_lr = self.lr.assign(self.initial_lr)

    def initialize(self, sess):
        sess.run(self.delta.initializer)
        sess.run(self.lr.initializer)
        
    def generate(self, sess, images, labels, use_noise=True):
        if use_noise:
            alpha = self.max_epsilon * 0.5
            delta_init = alpha * np.sign(np.random.randn(self.batchsize,299,299,3)).astype(np.float32)
        else:
            delta_init = np.zeros((self.batchsize,299,299,3), dtype=np.float32)
            
        sess.run(self.assign_delta, feed_dict={self.delta_to_assign: delta_init})
        sess.run(self.reset_lr)
        
        lower_bounds = np.maximum(-1 - images, -self.max_epsilon)
        upper_bounds = np.minimum(1 - images, self.max_epsilon)
        
        for _ in range(self.maxiter):
            logit, _,_  = sess.run([self.pred, self.train_op, self.update_lr], feed_dict={self.x_input:images,
                                                              self.y_input:labels,
                                                              self.lower_bounds:lower_bounds,
                                                              self.upper_bounds:upper_bounds})
	return images + sess.run(self.delta)
