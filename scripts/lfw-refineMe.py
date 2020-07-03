# This code fills superpixels by scribbling over the image with a given labeled color.
# It requires all jpg faces storaged in the same folder and the .dat super-pixels in the same LFW format.
# R. Redondo, Eurecat 2019 (c).

import numpy as np
import operator
import cv2
import os
import sys

resize = 3
pointer = (-1,-1)
isDrawing = False
radius = 10
category = 1
show_original = False

label_colors = [
	(  0,  0,  0),
	(  0,255,  0),
	(  0,  0,255),
	(255,255,  0),
	(255,  0,  0),
	(255,  0,255)]

label_names = [
	"eraser",
	"skin",
	"hair",
	"beard-mustache",
	"sunglasses",
	"wearable"]
 
def onClick(event,x,y,flags,param):

	global isDrawing, mode, radius, category, super_scribbles, pointer

	pointer = (int(x/resize), int(y/resize))

	if event == cv2.EVENT_LBUTTONDOWN:

		isDrawing = True

	elif event == cv2.EVENT_LBUTTONUP:

		isDrawing = False

	if isDrawing and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE):

		cv2.circle(labels, pointer, radius, label_colors[category], -1)

# ---------------------------------------------------------------------------------------

if len(sys.argv) < 3 or len(sys.argv) > 4:
	print("Usage: $ lfw-refineMe <faces_folder> <labels_folder> folder_per_person [optional boolean]")
	exit(0)

# faces_folder = '../Datasets/lfw-deepfunneled/'
# labels_folder = '../Datasets/elfw/elfw_01_basic/labels'

faces_folder 	= sys.argv[1]
labels_folder 	= sys.argv[2]

folder_per_person = False
if (len(sys.argv) == 4):
    folder_per_person = sys.argv[3] in '1tTrueYesy'

for labels_file in sorted(os.listdir(labels_folder)):

	if not labels_file.endswith(".png"):
		continue

	file_name = os.path.splitext(labels_file)[0]
	person_name = file_name[:-5]
	labels = cv2.imread(os.path.join(labels_folder, labels_file))

	if folder_per_person:
		path_name = os.path.join(os.path.join(faces_folder, person_name), file_name + '.jpg')
	else:
		path_name = os.path.join(faces_folder, file_name + '.jpg')

	if not os.path.exists(path_name):
		print('File not found: %s' % path_name)
		continue

	face = cv2.imread(path_name)

	print('Editing ' + '\033[1m' + labels_file + '\033[0m' + "...")

	# Mouse events callback
	cv2.namedWindow(file_name)
	cv2.setMouseCallback(file_name, onClick)

	# Defaults
	radius = 3
	category = 1

	while True:

		# Key handlers
		k = cv2.waitKey(1) & 0xFF
		if k >= 48 and k <= 53:
			category = k - 48
		elif k == ord('e'):	
			category = 0
		elif k == ord('q'):
			radius = min(radius + 2, 16)
		elif k == ord('a'):
			radius = max(radius - 2, 1)
		elif k == ord('x'):
			show_original = not show_original
		elif k == 32:
			if radius < 10:
				radius = 16
			else:
			 	radius = 1
		elif k == 13 or k == 10 or k == 141:
			break
		elif k == 27:
			exit(0)

		# Compositing
		alpha = 0.12
		face_canvas = face.copy()

		if not show_original:
			face_canvas[labels != 0] = face_canvas[labels != 0] * alpha + labels[labels != 0] * (1-alpha)
			cv2.circle(face_canvas, pointer, radius, label_colors[category], -1)

		vis = np.concatenate((face_canvas, labels), axis=1)
		vis = cv2.resize(vis, (vis.shape[1] * resize, vis.shape[0] * resize), interpolation = cv2.INTER_NEAREST) 

		# Info
		font_size = 0.6
		font_thickness = 2
		hstep = 25
		info = "Label (0-5,e): "
		cv2.putText(vis, info, (10, hstep * 1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255)) 
		info = "               " + label_names[category]
		cv2.putText(vis, info, (10, hstep * 1), cv2.FONT_HERSHEY_SIMPLEX, font_size, label_colors[category], font_thickness) 
		info = "Stroke (q-a,space): " + str(radius)
		cv2.putText(vis, info, (10, hstep * 2), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255))
		info = "Show Original (x)"
		cv2.putText(vis, info, (10, hstep * 3), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255))
		info = "Save and give me more (enter)"
		cv2.putText(vis, info, (10, hstep * 4), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255)) 		
		info = "Exit (esc)"
		cv2.putText(vis, info, (10, hstep * 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255)) 

		cv2.imshow(file_name, vis)

	cv2.destroyWindow(file_name)

	# Save output
	overwrite_file = os.path.join(labels_folder, labels_file)
	cv2.imwrite(overwrite_file, labels)
	print("Labels saved in " + overwrite_file)

cv2.destroyAllWindows()
