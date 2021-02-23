# This script goes over a folder of labels and finds those non-extended, i.e. background, skin, and hair only.
# R. Redondo (c) Eurecat 2021

import os
import sys
import numpy as np
import cv2
from shutil import copyfile

if len(sys.argv) != 4:
	print("Usage: $ elfw-findNonExtended <faces_folder> <labels_folder> <output_folder>")
	exit(0)

faces_path  = sys.argv[1]   # '../elfw/elfw_Baseline/faces'
labels_path = sys.argv[2]   # '../elfw/elfw_Baseline/labels'
if not os.path.exists(labels_path):
    print('Input path not found ' + labels_path)
    exit(-1)

output_path = sys.argv[3]   
output_path_faces  = os.path.join(output_path, 'faces')
output_path_labels = os.path.join(output_path, 'labels')

if not os.path.exists(output_path):
    os.mkdir(output_path)
if not os.path.exists(output_path_faces):
    os.mkdir(output_path_faces)
if not os.path.exists(output_path_labels):
    os.mkdir(output_path_labels)    

label_colors = [
    (0, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255)]
    # (255, 255, 0)]

grouped_per_name = True
non_extended_files = []

print('This will take a while...')
files = os.listdir(labels_path)

for file in files:

    file_path = os.path.join(labels_path, file)
    image = cv2.imread(file_path)
    # print(file_path)

    image_size = image.shape
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    extended = False

    for c in range(0, len(label_colors)):

        mask = (r == label_colors[c][0]) & (g == label_colors[c][1]) & (b == label_colors[c][2])
        pixels = np.sum(mask)

        if pixels and c > 2:
            extended = True

    if extended:
        continue

    non_extended_files.append(file)

    # Copy labels
    copyfile(file_path, os.path.join(output_path_labels, file))

    # Copy faces
    face_file = os.path.splitext(file)[0] + '.jpg'
    if grouped_per_name:
        face_grouped_file = os.path.join(face_file[:-9], face_file)
    else:
        face_grouped_file = ''
    copyfile(os.path.join(faces_path, face_grouped_file), os.path.join(output_path_faces, face_file))

    # cv2.imshow('Labels', image)
    # cv2.waitKey(0)

print(non_extended_files)
print('Number of faces without extended categories: {}'.format(len(non_extended_files)))