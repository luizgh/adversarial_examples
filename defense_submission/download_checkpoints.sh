#!/bin/bash
#
# Scripts which download checkpoints for provided models.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download inception v3 checkpoint for fgsm attack.
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz

wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
tar -xvzf inception_v4_2016_09_09.tar.gz

wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
tar xvf inception_resnet_v2_2016_08_30.tar.gz
mv inception_resnet_v2_2016_08_30.ckpt inception_resnet_v2.ckpt

wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
tar xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz

wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz

wget https://storage.googleapis.com/luizgh_data/tf-densenet169.tar.gz
tar xvf tf-densenet169.tar.gz

rm inception_v3_2016_08_28.tar.gz
rm inception_v4_2016_09_09.tar.gz
rm inception_resnet_v2_2016_08_30.tar.gz
rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz
rm tf-densenet169.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz
