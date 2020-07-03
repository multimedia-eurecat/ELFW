# This script takes all non-ground-truth-labeled images in LFW and copies them apart.
# Rafael Redondo - 2019

import os
import shutil

labels = os.listdir('../Datasets/parts/parts_lfw_funneled_gt_images/')
labels = [label.replace('.ppm','') for label in labels]
#print labels

output_folder = '../Datasets/lfw-deepfunneled-discarded/'
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

faces_folder = '../Datasets/lfw-deepfunneled/'

for person in os.listdir(faces_folder):

	if not os.path.isdir(faces_folder + person):
		continue

	for face_file in os.listdir(faces_folder + person):

		if not face_file.endswith(".jpg"):
			continue

		name = os.path.splitext(face_file)[0]

		if not any(name in s for s in labels):

			src_file = faces_folder  + person + "/" + face_file
			dst_file = output_folder 		  + "/" + face_file
			shutil.copyfile(src_file, dst_file)  

			print "Copied to " + output_folder + " the file " + src_file