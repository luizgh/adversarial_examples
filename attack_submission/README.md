# Non-targeted attack

Usage:

First download the checkpoints (6 models, ~1GB):
```
./download_checkpoints.sh #do
```

Run it:

```
./run_attack.sh <input_folder> <output_folder> <eps>
```

Where <input_folder> contains images in the .png format and <eps> is the maximum adversarial noise (maximum L_infinity norm of the adversarial noise delta).

## About this attack

We used a sequence of a few FGSM steps, followed by Projected Gradient Descent to minimize the logits (unormalized log-probabilities) of the correct class. We attack an ensemble of 6 models: inception_v3, inception_v4, inception_resnet_v2, densenet, adversarily trained inception_resnet_v2 and inception v3. For more information see: https://github.com/luizgh/adversarial_examples

