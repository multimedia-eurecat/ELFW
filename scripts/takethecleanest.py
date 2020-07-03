# This script takes all non-ground-truth-labeled images in LFW and copies them apart.
# Rafael Redondo - 2019

import os
import sys
import shutil

if len(sys.argv) != 3:
	print "Usage: $ takethecleanest <filelist> <output folder>"
	exit(0)

filelist = sys.argv[1]
f = open(filelist,"r")
labels = []
for line in f:
	labels.append(line)

output_folder = sys.argv[2]

if not os.path.exists(output_folder):
	os.mkdir(output_folder)

faces_folder = '../Datasets/lfw-deepfunneled-bagoffaces/all/'
labels_folder = '../Datasets/lfw-original_from_parts/'

#print labels 

for file in os.listdir(labels_folder):

	if not file.endswith(".jpg"):
		continue

	if not any(file in s for s in labels):

		src_file = faces_folder  + '/' + file
		dst_file = output_folder + '/' + file
		shutil.copyfile(src_file, dst_file)  
		print "Copied to " + output_folder + " the file " + src_file

