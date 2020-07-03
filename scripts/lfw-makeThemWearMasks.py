# This code augments the Labeled Faces in the Wild dataset with Masks!
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

# Make him wear it
def objectOverlay(canvas, item, reference_distance, reference_center, labels, item_type):

	obj = item.copy()

	# Item size adjustment
	resize_factor = ( reference_distance ) / (obj.shape[1] * 0.25)
	new_size = np.array([int(obj.shape[1] * resize_factor), int(obj.shape[0] * resize_factor)])
	new_size = np.array([new_size[0] + new_size[0] % 2, new_size[1] + new_size[1] % 2])
	obj = cv2.resize(obj, tuple(new_size))
	yc, xc = [int(reference_center[1] - 0.5 * obj.shape[0]), int(reference_center[0] - 0.5 * obj.shape[1])]
	b, g, r, a = cv2.split(obj)
	a3 = cv2.merge((a,a,a))
	obj = cv2.merge((b,g,r))

	# Margin crops
	left_top = np.array([ max(xc,0), max(yc,0)])
	right_bottom = np.array([ min(xc + obj.shape[1],canvas.shape[1]), min(yc + obj.shape[0],canvas.shape[0])])
	left_top_item = np.array([left_top[0]-xc,left_top[1]-yc])
	right_bottom_item = right_bottom - left_top + left_top_item
	a3 = a3[left_top_item[1]:right_bottom_item[1], left_top_item[0]:right_bottom_item[0]]
	obj = obj[left_top_item[1]:right_bottom_item[1], left_top_item[0]:right_bottom_item[0]]
	canvas_crop = canvas[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0],:]
	labels_crop = labels[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0],:]

	# Blending
	canvas_crop[a3>0] = obj[a3>0] * 0.92 + canvas_crop[a3>0] * 0.08
	t = label_names.index(item_type)
	lb, lg, lr = cv2.split(labels_crop)
	lb[a>0] = label_colors[t][0]
	lg[a>0] = label_colors[t][1]
	lr[a>0] = label_colors[t][2]
	labels[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0],:] = cv2.merge((lb,lg,lr))

#----------------------------------------------------------------------------------------------------

if len(sys.argv) != 5:
	print("Usage: $ lfw-makeThemWearMasks <faces folder> <labels folder> <wearables folder> <output folder>")
	exit(0)

faces_folder	 = sys.argv[1]
labels_folder	 = sys.argv[2]
wearables_folder = sys.argv[3]
output_folder	 = sys.argv[4]

output_folder_faces  = os.path.join(output_folder, 'faces')
output_folder_labels = os.path.join(output_folder, 'labels')
#output_folder_debug  = os.path.join(output_folder, 'debug')

if not os.path.exists(output_folder):
	os.mkdir(output_folder)
if not os.path.exists(output_folder_faces):
	os.mkdir(output_folder_faces)	
if not os.path.exists(output_folder_labels):
	os.mkdir(output_folder_labels)	
# if not os.path.exists(output_folder_debug):
# 	os.mkdir(output_folder_debug)	

# Not sure if the following will work with all opencv installation type
# I'm currently working with a virtual environment in which I have installed opencv 4.1.0 using pip
# The opencv location is the following ('env' is the virtual environment name):
# "/home/jaume.gibert/Code/facesinthewild/env/lib/python3.5/site-packages/cv2/"
haar_folder 	= os.path.join(os.path.dirname(cv2.__file__), 'data')
haar_face_ddbb 	= os.path.join(haar_folder, "haarcascade_frontalface_default.xml")
haar_eye_ddbb 	= os.path.join(haar_folder, "haarcascade_eye.xml")
haar_mouth_ddbb = os.path.join(haar_folder, "haarcascade_smile.xml")


print('\n' + bcolors.BOLD + 'Initiating Haar detector from ' + haar_folder + bcolors.ENDC)

face_cascade 	= cv2.CascadeClassifier()
if not face_cascade.load(haar_face_ddbb):
	print('--(!)Error loading face cascade')
	exit(0)

eye_cascade 	= cv2.CascadeClassifier()
if not eye_cascade.load(haar_eye_ddbb):
	print('--(!)Error loading eye cascade')
	exit(0)

mouth_cascade 	= cv2.CascadeClassifier()
if not mouth_cascade.load(haar_mouth_ddbb):
	print('--(!)Error loading mouth cascade')
	exit(0)

print(bcolors.GREEN + 'DONE!' + bcolors.ENDC)
print("")

#-----------------------------------------------------------------------------------------------------
# Keep masks around in a list of images 
masks = []
for wearable_file in os.listdir(wearables_folder):

	if not wearable_file.endswith(".png"):
		continue

	if fnmatch.fnmatch(wearable_file, '*mask*'):
		img = cv2.imread(wearables_folder + wearable_file, cv2.IMREAD_UNCHANGED)
		masks.append([img, os.path.splitext(wearable_file)[0]])


#-----------------------------------------------------------------------------------------------------
# For each image, look for the face and, if it does not already include sunglasses, paste some 
# over the (detected) eyes

counter_all_images     = 0
counter_no_jpg         = 0
counter_with_glasses   = 0
counter_no_face        = 0
counter_multiple_faces = 0
counter_no_eyes        = 0
counter_saved_images   = 0

N = len(os.listdir(faces_folder))

for n, face_file in enumerate(os.listdir(faces_folder)):

	counter_all_images += 1
	base_name = os.path.splitext(face_file)[0]

	# Print the image number and name
	if not n:
		sys.stdout.flush()
		print("")
	sys.stdout.write('\x1b[1A')
	sys.stdout.write('\x1b[2K')
	print(bcolors.BLUE + "["+ str(n).zfill(4) +"/"+ str(N) +"] " + base_name + bcolors.ENDC)

	if not face_file.endswith(".jpg"):
		counter_no_jpg += 1
		continue

#	# Use this to debug for a specific image or images...
#	if not fnmatch.fnmatch(face_file, '*Amer_al*'):
#		continue

	# Before doing anything, we should check whether the original image contains already some 
	# pixels labeled as sunglasses. If so, we will not augment this image

	# Load labels image
	labels = cv2.imread(os.path.join(labels_folder, base_name+'.png'))
#
#	# Build up a mask for the sunglasses class
#	sunglasses_color = label_colors[4]
#	mask = np.ones((labels.shape[0],labels.shape[1]))
#	for c in [0,1,2]:
#		mask_c = np.zeros((labels.shape[0],labels.shape[1]))
#		index = (labels[:,:,c] == sunglasses_color[c])
#		mask_c[index] = 1
#		mask = mask * mask_c
#
#	if np.sum(mask)>0:
#		#print(bcolors.BLUE + "Already has sunglasses: " + base_name + bcolors.ENDC)
#		#cv2.imwrite(output_folder_debug + base_name + '.ppm', labels)
#		counter_with_glasses += 1
#		continue

	# Face image
	image = cv2.imread(os.path.join(faces_folder, face_file))
	
	# Face pre-processing for detection of face and eyes
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray)
	ih, iw = gray.shape
	center = [iw*0.5, ih*0.5]

	# Detect faces in image
	faces = face_cascade.detectMultiScale(gray, 1.1, 6)
	if faces is None: 
		counter_no_face += 1
		continue
	#print(bcolors.YELLOW + "  -- Number of faces detected: " + str(len(faces)) + bcolors.ENDC)

	# When there are multiple detections, it is hard to tell which is the proper one
	# since we have lots of images to augment, we will discard these cases
	if len(faces)>1:
		counter_multiple_faces += 1
		continue
		
	# Put sunglasses on the different detected face - actually we only have one face, 
	# since other cases have been discarded
	for face_id, (x,y,w,h) in enumerate(faces):
		roi_gray  = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		#cv2.line(image,(0,125),(250,125),(255, 0, 0), 1)

		# Eyes detection on the current face
		eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6)
		num_eyes  = 0
		right_eye = 0
		left_eye  = 0
		middle_eye = np.array([0,0])

		for (ex, ey, ew, eh) in eyes:
			eye_center = [x + ex + ew * 0.5, y + ey + eh * 0.5]
			if eye_center[1] < center[0]:
				# cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
				middle_eye = middle_eye + eye_center
				if eye_center[0] > center[0]:
					left_eye = eye_center
				else :
					right_eye = eye_center
				num_eyes = num_eyes + 1

		if num_eyes != 2 or not left_eye or not right_eye:
			counter_no_eyes += 1
			continue

		middle_eye = middle_eye / num_eyes

		# mouth detection
		mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 7)
		if mouths is None: 
			continue
	
		# When there are multiple detections, it is hard to tell which is the proper one
		# since we have lots of images to augment, we will discard these cases
		if len(mouths)>1:
			counter_multiple_faces += 1
			continue

		for (mx, my, mw, mh) in mouths:
			mouth_center = [x + mx + mw * 0.5, y + my + mh * 0.5]
			if not mouth_center[1] > center[1]:
				break

		# Paste mask over the mouth 
		# Use only a random number of the available: shuffle the list and take the k first
		shuffle(masks)
		for i in range(10):
			
			# create copies so we don't keep pasting glasses on the same image all the time
			im = image.copy()
			lb = labels.copy()

			# augmentation id for storing the file
			M      = masks[i][0]
			aug_id = masks[i][1]
			#aug_id = str(i).zfill(4)

			# overlay the sunglasses
			objectOverlay(im, M, left_eye[0] - middle_eye[0], mouth_center, lb, "mouth-mask" )


			# save image and labels
			augmented_face_file   = os.path.join(output_folder_faces,  base_name+'_'+aug_id+'.jpg')
			augmented_labels_file = os.path.join(output_folder_labels, base_name+'_'+aug_id+'.png')
			if not os.path.isfile(augmented_face_file):
				cv2.imwrite(augmented_face_file,   im)
				cv2.imwrite(augmented_labels_file, lb)
				counter_saved_images += 1
			else:
				print(bcolors.RED + "File already exists: " + augmented_face_file + bcolors.ENDC)
		
print("\n" + bcolors.RED  + "Total number of files .... " + bcolors.ENDC + str(counter_all_images))
print("\n" + bcolors.BOLD + "No jpg images ............ " + bcolors.ENDC + str(counter_no_jpg))
print(       bcolors.BOLD + "With sunglasses .......... " + bcolors.ENDC + str(counter_with_glasses))
print(       bcolors.BOLD + "No face detected ......... " + bcolors.ENDC + str(counter_no_face))
print(       bcolors.BOLD + "Several faces detected ... " + bcolors.ENDC + str(counter_multiple_faces))
print(       bcolors.BOLD + "No eyes detected ......... " + bcolors.ENDC + str(counter_no_eyes))
print(       bcolors.BOLD + "Saved images ............. " + bcolors.ENDC + str(counter_saved_images))
print("\n")

cv2.destroyAllWindows()	