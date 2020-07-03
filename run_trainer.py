from __future__ import division
import sys
import math, time
import argparse
from elfw import *
from utils import *
from trainer import TrainVal

# Rafael Redondo, Jaume Gibert - Eurecat (c) 2019
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Arguments

ap = argparse.ArgumentParser(prog="trainer_elfw.py")

ap.add_argument("-Vs",  
                "--validation_set_id", 
                type=int, 
                help="Id of the validation set", # check elfw.py for details
                default=0)

ap.add_argument("-S",  
                "--synthetic_augmentation_rate", 
                type=str, 
                help="Rate for synthetic augmentation.", 
                default=0)

ap.add_argument("-St", 
                "--synthetic_augmentation_types", 
                type=str, 
                help="Configuration of which synthetic objects are used: input must be a comma-separated string of integers such as 0,1,2", 
                default=None)

ap.add_argument("-e", 
                "--excluded_classes", 
                type=str, 
                help="List of classes that won't be used for training nor validation: input must be a comma-separated string of integers such as 0,1,2", 
                default=None)

ap.add_argument("-M",  
                "--model", 
                type=str, 
                help="Segmentation model: fcn, gcn or deeplab.", 
                default="fcn")

ap.add_argument("-bs", 
                "--batch_size", 
                type=str, 
                help="The batch size", 
                default=1)

ap.add_argument("-K",  
                "--checkpoints_path", 
                type=str, 
                help="Path to store the checkpoints", 
                default="/media/hd/elfw/checkpoints")

ap.add_argument("-R",  
                "--resume_checkpoint",
                type=str, 
                help="Resumes training at this checkpoint", 
                default=None)

args = vars(ap.parse_args())

# -------------------------------------------------------------------------
# CUDA availability

if not torch.cuda.is_available():
    print("Error: CUDA not available")
    exit(0)

# -------------------------------------------------------------------------
# Command line arguments

batch_size     = int(args['batch_size'])
Vs             = args['validation_set_id']      # Index of the validation set file, see elfw.py.
K              = args['checkpoints_path']
M              = args['model']
R              = args['resume_checkpoint']
S              = float(args['synthetic_augmentation_rate'])

St = args['synthetic_augmentation_types']
St = St if not St else list(map(int, St.split(',')))

if S and not St:
    sys.exit("check your parameters: if the augmentation ratio (-S) is positive, there should be at least one augmentation type (-St)")
if S==0 and St:
    sys.exit("check your parameters: if the augmentation ratio (-S) is zero, you should specify the augmentation types (-St)")

e = args['excluded_classes']
e = e if not e else list(map(int, e.split(',')))

# -------------------------------------------------------------------------
# Some other hyperparameters

gcn_levels     = 3        # Number of GCN levels, typically 3 for 256x256 and 4 for 512x512 image sizes
max_epochs     = 301      # Maximum number of epochs 
lr             = 1E-3     # Learning rate
lr_decay       = 0.2      # Learning rate decay factor
w_decay        = 5E-4     # Weight decay, typically [5e-4]
momentum       = 0.99     # Momentum, typically [0.9-0.99]
lr_milestones  = [35,90,180] # lr milestones for a multistep lr scheduler
augment        = True     # random transformations for data augmentation

# -------------------------------------------------------------------------
# Train and Validation data sets and data loaders

ELFW_train = ELFWDataSet(split="train",
                         valset=Vs,
                         random_transform=augment,
                         synth_augmen_types=St,
                         synth_augmen_ratio=S,
                         compute_class_weights=True,
                         excluded_classes=e)

trainLoader = data.DataLoader(ELFW_train, 
                              batch_size=batch_size,
                              num_workers=16,
                              shuffle=True,
                              pin_memory=True)

# The VALIDATION dataset and the corresponding data loader
ELFW_validation = ELFWDataSet(split="validation",
                              valset=Vs,
                              excluded_classes=e)

valLoader = data.DataLoader(ELFW_validation,
                            batch_size=batch_size,
                            num_workers=16,
                            shuffle=False,
                            pin_memory=True)

start_time = time.time()

TrainVal(trainLoader,
         valLoader,
         max_epochs,
         lr,
         lr_decay,
         lr_milestones,
         w_decay,
         momentum,
         augment,
         S,
         K,
         R,
         M,
         gcn_levels)

elapsed_time = time.time() - start_time
hours   = int(math.floor(elapsed_time / 3600))
minutes = int(math.floor(elapsed_time / 60 - hours * 60))
seconds = int(math.floor(elapsed_time - hours * 3600 - minutes * 60))
print('Training finished in \033[1m' + str(hours) + 'h ' + str(minutes) + 'm ' + str(seconds) + 's\033[0m')

print("\n")
