from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import tensorflow as tf

class box_constraint_adversarial_batches:
    def __init__(self, model, max_epsilon, batchsize, max_fun_iter=5, max_ls=20, targeted=False):
        self.batchsize = batchsize
        self.max_fun_iter = max_fun_iter

        self.x_input = tf.placeholder(tf.float32, shape=(None,299,299,3))
        self.y_input = tf.placeholder(tf.int32)

        self.probs = model.predict(self.x_input)

        row_indices = tf.range(self.batchsize)
        indices = tf.transpose([row_indices, self.y_input])
        self.pred = tf.gather_nd(self.probs, indices)

        self.grad = tf.gradients(tf.reduce_sum(self.pred), self.x_input)
        self.max_epsilon = max_epsilon
        self.targeted = targeted
        self.maxls = max_ls


    def generate(self, sess, img, correct_or_target_class, use_noise):
        lower_bounds = np.maximum(-1 - img, -self.max_epsilon).reshape(-1)
        upper_bounds = np.minimum(1 - img, self.max_epsilon).reshape(-1)
        bounds = list(zip(lower_bounds, upper_bounds))

        def func(delta):
            attack_img = img + delta.reshape(self.batchsize, 299, 299, 3).astype(np.float32)
            logit, gradients = sess.run([self.pred] + self.grad,
                                        feed_dict={self.x_input: attack_img,
                                                   self.y_input: correct_or_target_class})
            if self.targeted:
                # Multiply by -1 since we want to maximize it.
                return -1 * np.mean(logit), -1 * gradients.reshape(-1).astype(np.float)
            else:
                return np.mean(logit), gradients.reshape(-1).astype(np.float)

        if use_noise:
            alpha = self.max_epsilon * 0.5
            x0 = alpha * np.sign(np.random.random(self.batchsize*299*299*3))
        else:
            x0 = np.zeros(self.batchsize*299*299*3),
        delta_best, f, d = fmin_l_bfgs_b(func=func,
                                   x0=x0, 
                                   bounds=bounds,
                                   maxfun=self.max_fun_iter,
                                   maxls=self.maxls)
        return img + delta_best.reshape(-1, 299, 299, 3).astype(np.float32)
