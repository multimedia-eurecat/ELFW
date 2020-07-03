# This script takes all non-ground-truth-labeled images in LFW and copies them apart.
# Rafael Redondo - 2019

import os
import sys
import shutil

if len(sys.argv) != 4:
	print("Usage: $ takefromfile <filelist> <faces folder> <output folder>")
	exit(0)

filelist = sys.argv[1]
f = open(filelist,"r")
targets = []
for line in f:
	targets.append(line)

# faces_folder = '../Datasets/lfw-deepfunneled/'
faces_folder = sys.argv[2]
output_folder = sys.argv[3]

if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# print(targets)
 
for person in os.listdir(faces_folder):

	person_path = os.path.join(faces_folder, person)

	if not os.path.isdir(person_path):
		continue

	for face_file in os.listdir(person_path):

		if not face_file.endswith(".jpg"):
			continue

		name = os.path.splitext(face_file)[0] + os.path.splitext(face_file)[1]

		if any(name in s for s in targets):
			src_file = os.path.join(person_path, face_file)
			dst_file = os.path.join(output_folder, face_file)
			shutil.copyfile(src_file, dst_file)  
			print "Copied to " + output_folder + " the file " + src_file

