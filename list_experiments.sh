#!/bin/bash

# -M is the model
# options are fcn, deeplab and gcn
# don't use the gcn option. we haven't test it properly

# -e are the excluded classes
# this will typically be only #6 (mouth-mask) or none

# -bs is the model batch size

# -Vs is the validation set ID
# depending on the experiment we will either use a validation set or another
# in particular, if we are testing for (only) sunglasses augmentation
# we have a validation set with half the images with sunglasses from ELFW. etc.
# 0 is for sunglasses
# 1 is for hands
# 2 is a general validation set with random images

# -S synthetic augmentation ratio
# it has to be positive and is the ratio of training images that are used from the
# augmentation folders (next parameter)
# if there are 90 images in the train set and we set -S 0.1 we are asking for 90*0.1=9 (synthethically) augmented images
# these images are taken from the augmentation folders uniformly, this is, if there is 1 folder, the 9 are taken from it
# if there are 2 folders, we take int(9/2) from 1 and int(9/2) from the other, 
# if there are 3 folders, we take int(9/3) from 1, int(9/3) from the second one and int(9/3) from the last one 

# -St is the augmentation types configuration
# 0 is for Sunglasses
# 1 is for Hands
# 2 is for Masks
# we can combine them as we with, for example, -St 0,1,2 or -St 0 or -St 1,2
# be sure you give the numbers in increasing ordre. nothing bad will happen but the -St 1,2 and -St 2,1 experiments are
# technically the same although it won't be handled

##################################
# LIST OF EXPERIMENTS
#
# Sunglasses augmentations:
#	* St will only be "-St 0"
#	* we exclude the mouth-mask class, so always "-e 6"
#	* the model is either fcn or deeplab
#	* batch size will depend on the GPU capacity. for the fcn is 16, deeplab is a little lighter so 16 will be ok
#	* Validation set will be "-Vs 0"
# 	* Different augmentation ratios "-S x" for x in [0, 0.25, 0.5, 0.75, 1] (for -S 0, -St is none)
#
# Hands augmentations:
#	* St will only be "-St 1"
#	* we exclude the mouth-mask class, so always "-e 6"
#	* the model is either fcn or deeplab
#	* batch size will depend on the GPU capacity. for the fcn is 16, deeplab is a little lighter so 16 will be ok
#	* Validation set will be "-Vs 1"
# 	* Different augmentation ratios "-S x" for x in [0, 0.25, 0.5, 0.75, 1] (for -S 0, -St is none)
#
# All types of augmentations:
#	* St will be "-St 0,1,2"
#	* we have all classes so -e is none: problem here, evaluation will consider mouth-masks although there is none in the validation set
#	* the model is either fcn or deeplab
#	* batch size will depend on the GPU capacity. for the fcn is 16, deeplab is a little lighter so 16 will be ok
#	* Validation set will be "-Vs 2"
# 	* Different augmentation ratios "-S x" for x in [0, 0.25, 0.5, 0.75, 1] (for -S 0, -St is none)


# Sunglasses augmentations
# these first 4 exps will have different names
[2.55] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 #
[3.21] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 -St 0 -S 0.25 # 
[3.85] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 -St 0 -S 0.5 #
[5.13] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 -St 0 -S 1.0 #

python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0 -St 0 -S 0.25
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0 -St 0 -S 0.5
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0 -St 0 -S 1.0

# Hands augmentations
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1 -St 1 -S 0.25
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1 -St 1 -S 0.5
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1 -St 1 -S 1.0
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1 -St 1 -S 0.25
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1 -St 1 -S 0.5
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1 -St 1 -S 1.0

# All augmentations - mouth-masks included (for webcam validation - results on val set won't be reported!)
# results will only be qualitative so we only train a single model
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -Vs 2 -St 0,1,2 -S 0.5

# All augmentations - mouth-masks excluded (for validating that several augmentations also help)
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.25
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.5
python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2 -St 0,1 -S 1.0
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.25
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.5
python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2 -St 0,1 -S 1.0


# Sort them all! 

# @ GPU0
[3.85] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.5
[3.21] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1 -St 1 -S 0.25
[5.13] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 -St 0 -S 1.0 #
[4.84] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0 -St 0 -S 0.25
[5.81] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0 -St 0 -S 0.5
[5.81] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.5
#Total time = 3.85+3.21+4.84+5.81+5.81+5.13=28.65

# @ GPU1
[3.21] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 -St 0 -S 0.25 # 
[5.13] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1 -St 1 -S 1.0
[3.85] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1 -St 1 -S 0.5
[7.74] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0 -St 0 -S 1.0
[3.87] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1
[4.84] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1 -St 1 -S 0.25
#Total time = 5.13+3.85+7.74+3.87+4.84+3.21=28.64

# @ GPU2
[2.55] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 
[2.57] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2
[3.21] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.25
[5.13] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 1
[5.81] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1 -St 1 -S 0.5
[7.74] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 1 -St 1 -S 1.0
[3.87] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2
# Total time = 2.57+3.21+5.13+5.81+7.74+3.87+2.55=30.88

# @ GPU3
[3.85] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 0 -St 0 -S 0.5 
[5.13] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -e 6 -Vs 2 -St 0,1 -S 1.0
[3.85] python pytorch-segmentation/run_trainer.py -M fcn -bs 16 -Vs 2 -St 0,1,2 -S 0.5
[3.87] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 0
[4.84] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2 -St 0,1 -S 0.25
[7.74] python pytorch-segmentation/run_trainer.py -M deeplab -bs 16 -e 6 -Vs 2 -St 0,1 -S 1.0
# Total time = 5.13+3.85+3.87+4.84+7.74+3.85=29.28







