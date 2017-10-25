# Box-constrained attacks (adversarial examples) in Tensorflow

This repository contains the code for the targeted and non-targeted attacks used by team Placeholder ([Luiz Gustavo Hafemann](https://github.com/luizgh) and [Le Thanh nguyen-meidine](https://github.com/Natlem) for the NIPS 2017 adversarial attacks/defenses competition.

Our attacks were based on a formulation of the problem that minimizes the probability of the correct class (or maximize the probability of a target class) considering the distortion (L_infinity norm of the adversarial noise) as a hard constraint. For this we used two algorithms: L-BFGS with box-constraints and projected Stochastic Gradient Descent. 

## Setup

For the attacks we used python 2 (which was recommended for the competition), but the refactored version of the attacks in this main folder work with either Python 2 or Python 3.

Requirements:

* numpy
* tensorflow (tested with versions 1.2 and 1.4)
* scipy (tested with version 0.19.1)

## Usage

We are making three attacks available:

* box_constrained_attack: Uses L-BFGS with box-constraints as optimizer
* pgd_attack: Uses projected SGD (Stochastic Grandient Descent) as optimizer
* step_pgd_attcK: Uses a mix of FGSM (Fast Gradient Sign Attack) and SGD. We found this to converge faster if there is a limit of only a few iterations (e.g. 10-15)

Example of usage:
```
pgd_attacker = pgd_attack.PGD_attack(model, 
                                     batch_shape, 
                                     max_epsilon=eps, 
                                     max_iter=max_iter, 
                                     targeted=False,
                                     initial_lr=1,
                                     lr_decay=0.99)

attack_img = pgd_attacker.generate(sess, imgs, pred, verbose=True)                                     
```

The parameters are explained in the docstring. For the method above:
   * model: Callable (function) that accepts an input tensor 
          and return the model logits (unormalized log probs)
   * batch_shape: Input shapes (tuple). 
          Usually: (batch_size, height, width, channels)
   * max_epsilon: Maximum L_inf norm for the adversarial example
   * max_iter: Maximum number of gradient descent iterations
   * targeted: Boolean: true for targeted attacks, false for non-targeted attacks
   * img_bounds: Tuple [min, max]: bounds of the image. Example: [0, 255] for
          a non-normalized image, [-1, 1] for inception models.
   * initial_lr: Initial Learning rate for the optimization
   * lr_decay: Learning rate decay (multiplied in the lr in each iteration)
   * rng: Random number generator 

For a more comprehensive example, please check the provided [ipython notebook](https://github.com/luizgh/adversarial_examples/blob/master/example.ipynb)

