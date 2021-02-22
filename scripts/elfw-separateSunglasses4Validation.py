# This code lists all images that have sunglasses labelled in the Labeled Faces in the Wild dataset
# J.Gibert based on code by R.Redondo, Eurecat 2019 (c).

import numpy as np
import sys
import cv2
import os
import fnmatch
import imutils
from random import randint
from random import random
from random import shuffle

label_colors = [
	(  0,  0,  0), # black      - background
	(  0,255,  0), # green      - skin
	(  0,  0,255), # red        - hair
	(255,255,  0), # light blue - beard-mustache
	(255,  0,  0), # blue       - sunglasses
	(255,  0,255), # pink       - wearable
	(  0,255,255)] # yellow     - mouth-mask 

label_names = [
	"background",
	"skin",
	"hair",
	"beard-mustache",
	"sunglasses",
	"wearable",
	"mouth-mask"]

class bcolors:
    PURPLE = '\033[95m'
    BLUE   = '\033[94m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    CYAN   = '\033[96m'
    ENDC   = '\033[0m'
    BOLD   = '\033[1m'
    CYAN   = '\033[96m'


#----------------------------------------------------------------------------------------------------

if len(sys.argv) != 3:
	print("Usage: $ elfw-separateSunglasses4Validation.py <labels folder> <output_file_location>")
	exit(0)

labels_folder      = sys.argv[1]
output_file_folder = sys.argv[2]

#-----------------------------------------------------------------------------------------------------
# For each image, check if it has sunglasses and print its name out in a file

N = len(os.listdir(labels_folder))
count = 0

f_with    = open(os.path.join(output_file_folder,"with_sunglasses.txt"), 'w')
f_without = open(os.path.join(output_file_folder,"without_sunglasses.txt"), 'w')

for n, face_file in enumerate(os.listdir(labels_folder)):

	base_name = os.path.splitext(face_file)[0]

	# Print the image number and name
	if not n:
		sys.stdout.flush()
		print("")
	sys.stdout.write('\x1b[1A')
	sys.stdout.write('\x1b[2K')
	print(bcolors.BLUE + "["+ str(n).zfill(4) +"/"+ str(N) +"] " + base_name + bcolors.ENDC)

	# Load labels image
	labels = cv2.imread(os.path.join(labels_folder, base_name+'.png'))

	# Build up a mask for the sunglasses class
	sunglasses_color = label_colors[4]
	mask = np.ones((labels.shape[0],labels.shape[1]))
	for c in [0,1,2]:
		mask_c = np.zeros((labels.shape[0],labels.shape[1]))
		index = (labels[:,:,c] == sunglasses_color[c])
		mask_c[index] = 1
		mask = mask * mask_c

	if np.sum(mask)>0:
		#print(bcolors.BLUE + "Already has sunglasses: " + base_name + bcolors.ENDC)
		count +=1
		f_with.write(base_name+"\n")
		cv2.imwrite(os.path.join('/media/jaume.gibert/Data/elfw/elfw_01_basic', 'with_sunglasses', base_name+'.png'), labels)
	else:
		f_without.write(base_name+"\n")

f_with.close()
f_without.close()
		
print("\n" + bcolors.RED  + "Total number of files .... " + bcolors.ENDC + str(N))
print(       bcolors.BOLD + "With sunglasses .......... " + bcolors.ENDC + str(count))
print("\n")

