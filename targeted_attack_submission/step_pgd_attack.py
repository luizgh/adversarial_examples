import tensorflow as tf
import numpy as np


class step_pgd_attack:
    """ Creates adversarial samples using a projected gradient descent attack
    """
    def __init__(self, model, 
                 batch_shape, 
                 max_epsilon, 
                 max_iter, 
                 targeted, 
                 alpha,
                 step_iter,
                 img_bounds=[-1, 1],
                 use_noise=True,
                 initial_lr=0.5, 
                 lr_decay=0.98,
                 rng = np.random.RandomState()):
        """ 
             model: Callable (function) that accepts an input tensor 
                    and return the model logits (unormalized log probs)
             batch_shape: Input shapes (tuple). 
                    Usually: [batch_size, height, width, channels]
             max_epsilon: Maximum L_inf norm for the adversarial example
             max_iter: Maximum number of gradient descent iterations
             targeted: Boolean: true for targeted attacks, false for non-targeted attacks
             img_bounds: Tuple [min, max]: bounds of the image. Example: [0, 255] for
                    a non-normalized image, [-1, 1] for inception models.
             initial_lr: Initial Learning rate for the optimization
             lr_decay: Learning rate decay (multiplied in the lr in each iteration)
             rng: Random number generator 
             
        """
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        self.y_input = tf.placeholder(tf.int32, shape=(batch_shape[0]))
        
        # Symbolic variables for the adversarial noise and image
        # Note: we use delta as a Variable (as oposed to placeholder)
        #       so it doesn't need to be moved from GPU->CPU on each
        #       iteration
        self.delta = tf.Variable(np.zeros(batch_shape, dtype=np.float32))
        adv_img = self.x_input + self.delta

        # Loss function 
        #  Gather the logits of the correct/target classes:
        row_indices = tf.range(batch_shape[0])
        indices = tf.transpose([row_indices, self.y_input])
        logits = model(adv_img)
        logits_correct_class = tf.gather_nd(logits, indices)
        self.loss = tf.reduce_mean(logits_correct_class)
        grad = tf.gradients(self.loss, self.delta)

        # Update rule: Projected gradient descent:
        self.lr = tf.Variable(initial_lr, dtype=tf.float32)  # learning rate

        self.lower_bounds = tf.placeholder(tf.float32, shape=batch_shape) # upper bound on delta
        self.upper_bounds = tf.placeholder(tf.float32, shape=batch_shape) # lower bound on delta
        
        if targeted:
            multiplier = 1. # For targeted attack: maximize logits of desired class
        else:
            multiplier = -1. # For non-targeted attack: minimize logits of correct class
            
        delta_new = self.delta + multiplier * self.lr * grad[0]
        
        # Project delta back to the valid region (given by the bounds)
        delta_new = tf.maximum(delta_new, self.lower_bounds)
        delta_new = tf.minimum(delta_new, self.upper_bounds)
        
        assign_op = tf.assign(self.delta, delta_new)
        update_lr = tf.assign(self.lr, self.lr * lr_decay)
        self.train_op = tf.group(assign_op, update_lr) # Training op: update delta and lr

        delta_new_iter = self.delta + multiplier * alpha * tf.sign(grad[0])
        delta_new_iter = tf.maximum(delta_new_iter, self.lower_bounds)
        delta_new_iter = tf.minimum(delta_new_iter, self.upper_bounds)
        
        self.train_op_iter = tf.assign(self.delta, delta_new_iter)

        
        # Operation to reset delta and the learning rate
        self.delta_to_assign = tf.placeholder(tf.float32)
        self.assign_delta = self.delta.assign(self.delta_to_assign)
        self.reset_lr = tf.assign(self.lr, initial_lr)
        
        # Keep track of the parameters:
        self.step_iter = step_iter
        self.max_iter = max_iter
        self.max_epsilon = max_epsilon
        self.batch_shape = batch_shape
        self.img_bounds = img_bounds
        self.use_noise = use_noise
        self.rng = rng

    def initialize(self, sess):
        """ Initializes the variables in the session (necessary if not running 
            the "tf.global_variables_initializer", such as when only pre-trained 
            models are used)
        """
        sess.run(self.delta.initializer)
        sess.run(self.lr.initializer)
        
    def generate(self, sess, images, labels_or_targets, verbose=False):
        """ Generates adversarial images
            sess: the tensorflow session
            images: a 4D tensor containing the original images
            labels_or_targets: for non-targeted attacks, the actual or predicted labels
                               for targeted attacks, the desired target classes for each image.
            
            returns: adv_images: a 4D tensor containing adversarial images
        """
        if self.use_noise:
            # Random starting step, from https://arxiv.org/abs/1705.07204
            alpha = self.max_epsilon * 0.5
            delta_init = alpha * np.sign(self.rng.normal(size=self.batch_shape)).astype(np.float32)
        else:
            # Or start from the original image (i.e. no perturbation)
            delta_init = np.zeros((batch_shape), dtype=np.float32)
            
        # Reset the values of delta and learning rate
        sess.run(self.assign_delta, feed_dict={self.delta_to_assign: delta_init})
        sess.run(self.reset_lr)
        
        # Calculate the bounds for the perturbation
        lower_bounds = np.maximum(self.img_bounds[0] - images, -self.max_epsilon)
        upper_bounds = np.minimum(self.img_bounds[1] - images, self.max_epsilon)
        
       
        for i in range(self.step_iter):
            l, _  = sess.run([self.loss, self.train_op_iter], 
                                 feed_dict={self.x_input:images,
                                            self.y_input:labels_or_targets,
                                            self.lower_bounds:lower_bounds,
                                            self.upper_bounds:upper_bounds})
            if verbose:
                print('Iter %d, loss: %.2f' % (i, l))
        for i in range(self.step_iter, self.max_iter):
            l, _  = sess.run([self.loss, self.train_op], 
                                 feed_dict={self.x_input:images,
                                            self.y_input:labels_or_targets,
                                            self.lower_bounds:lower_bounds,
                                            self.upper_bounds:upper_bounds})
            if verbose:
                print('Iter %d, loss: %.2f' % (i, l))
        return images + sess.run(self.delta)
