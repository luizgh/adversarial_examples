# Defense 

Usage:

First download the checkpoints (6 models, ~1GB):
```
./download_checkpoints.sh 
```

Run it:

```
./run_defense.sh <input_folder> <output_file>
```

Where <input_folder> contains images in the .png format

## About this defense

This is a simple defense that uses zoom, random crops and rotations of the input images. We use an ensemble of 6 models: inception_v3, inception_v4, inception_resnet_v2, densenet, adversarily trained inception_resnet_v2 and inception v3. 

