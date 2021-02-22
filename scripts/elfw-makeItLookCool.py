# [Deprecated, see augmentation scripts for each augmentation object (hands, sunglasses, and masks)]
#
# This code uses the Viola-Jones face detector to
# augment face images with synthetic sunglasses, hands and color patches.
#
# R. Redondo, Eurecat 2019 (c).

import numpy as np
import sys
import cv2
import os
import fnmatch
import imutils
from random import randint
from random import random

label_colors = [
	(  0,  0,  0),
	(  0,255,  0),
	(  0,  0,255),
	(255,255,  0),
	(255,  0,  0),
	(255,  0,255),
	(  0,255,255)]

label_names = [
	"background",
	"skin",
	"hair",
	"beard-mustache",
	"sunglasses",
	"wearable",
	"mouth-mask"]

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

if len(sys.argv) != 6:
	print("Usage: $ elfw-makeItLookCool <faces folder> <labels folder> <wearables folder> <hands folder> <output folder>")
	exit(0)

faces_folder 		= sys.argv[1] + '/'
labels_folder 		= sys.argv[2] + '/'
wearables_folder 	= sys.argv[3] + '/'
hands_folder 		= sys.argv[4] + '/'
output_folder 		= sys.argv[5] + '/'

# faces_folder = '../Datasets/lfw-deepfunneled/'
# output_folder = '../Datasets/lfw-deepfunneled-wearables/'

output_folder_faces  = output_folder + '/faces/'
output_folder_labels = output_folder + '/labels/'
output_folder_faces_occluded  = output_folder + '/faces_occluded/'
output_folder_labels_occluded = output_folder + '/labels_occluded/'

if not os.path.exists(output_folder):
	os.mkdir(output_folder)

if not os.path.exists(output_folder_faces):
	os.mkdir(output_folder_faces)	

if not os.path.exists(output_folder_labels):
	os.mkdir(output_folder_labels)	

if not os.path.exists(output_folder_faces_occluded):
	os.mkdir(output_folder_faces_occluded)	

if not os.path.exists(output_folder_labels_occluded):
	os.mkdir(output_folder_labels_occluded)

haar_folder 	= "facedetectors/opencv/haarcascades"
haar_face_ddbb 	= haar_folder + "/haarcascade_frontalface_default.xml"
haar_eye_ddbb 	= haar_folder + "/haarcascade_eye.xml"
haar_mouth_ddbb = haar_folder + "/haarcascade_mouth.xml"
print('\033[1m' + 'Initiating Haar detector from' + haar_folder + '\033[0m')
face_cascade 	= cv2.CascadeClassifier(haar_face_ddbb)
eye_cascade 	= cv2.CascadeClassifier(haar_eye_ddbb)
mouth_cascade 	= cv2.CascadeClassifier(haar_mouth_ddbb)

#-----------------------------------------------------------------------------------------------------
# Wearables

sunglasses = []
masks = []

for wearable_file in os.listdir(wearables_folder):

	if not wearable_file.endswith(".png"):
		continue

	img = cv2.imread(wearables_folder + wearable_file, cv2.IMREAD_UNCHANGED)

	if fnmatch.fnmatch(wearable_file, '*sunglasses*'):
		sunglasses.append(img)
	elif fnmatch.fnmatch(wearable_file, '*mask*'):
		masks.append(img)


#-----------------------------------------------------------------------------------------------------
# Occluders

hands = []

for hand_file in os.listdir(hands_folder):

	if not hand_file.endswith(".png"):
		continue

	img = cv2.imread(hands_folder + hand_file, cv2.IMREAD_UNCHANGED)
	hands.append(img)

#-----------------------------------------------------------------------------------------------------

for face_file in os.listdir(faces_folder):

	if not face_file.endswith(".jpg"):
		continue

	name = os.path.splitext(face_file)[0]

	# Face image
	image = cv2.imread(faces_folder + face_file)
	# b_channel, g_channel, r_channel = cv2.split(image)
	# alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
	# image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

	# Face pre-processing
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(gray)
	ih, iw = gray.shape
	center = [iw*0.5,ih*0.5]

	# Face detection
	faces = face_cascade.detectMultiScale(gray, 1.1, 6)
	if faces is None: 
		continue

	# Label image
	labels_name = os.path.splitext(face_file)[0]
	labels = cv2.imread(labels_folder + labels_name + '.ppm')

	# Change default background color
	default_background_color = (255,0,0)
	mask = np.ones((labels.shape[0],labels.shape[1]))
	for c in [0,1,2]:
		mask_c = np.zeros((labels.shape[0],labels.shape[1]))
		index = labels[:,:,c] == default_background_color[c]
		mask_c[index] = 1
		mask = mask * mask_c

	for c in [0,1,2]:
		labels[:,:,c] = labels[:,:,c] * (1-mask)	

	# Put objets on the face
	for (x,y,w,h) in faces:
		roi_gray  = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		#cv2.line(image,(0,125),(250,125),(255, 0, 0), 1)

		# Eyes
		eyes = eye_cascade.detectMultiScale(roi_gray,1.1, 6)
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
			continue

		middle_eye = middle_eye / num_eyes
		#cv2.circle(image, (int(left_eye[0]),int(left_eye[1])), 4, (0, 255, 255), 4)
		#cv2.circle(image, (int(right_eye[0]),int(right_eye[1])), 4, (0, 255, 255), 4)
		#cv2.circle(image, (int(middle_eye[0]),int(middle_eye[1])), 4, (0, 0, 255), 4)

		occluded = random() < 0.5
		has_object = False

		# Sunglasses
		if not occluded and random() < 0.8:

			i = randint(0, len(sunglasses)-1)
			objectOverlay(image, sunglasses[i], left_eye[0] - middle_eye[0], middle_eye, labels, "sunglasses" )
			has_object = True

		# Mouth mask
		if not occluded and (random() < 0.2 or not has_object):

			mouths = mouth_cascade.detectMultiScale(roi_gray, 1.3, 7)

			for (mx, my, mw, mh) in mouths:
				mouth_center = [x + mx + mw * 0.5, y + my + mh * 0.5]
				if mouth_center[1] > center[1]:
					#cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (255, 255, 0), 2)
					i = randint(0, len(masks)-1)
					objectOverlay(image, masks[i], left_eye[0] - middle_eye[0], mouth_center, labels, "mouth-mask")
					break

		if occluded:

			# Hands
			if random() < 0.5:

				i = randint(0, len(hands)-1)
				hand_hsv = cv2.cvtColor(hands[i], cv2.COLOR_BGR2HSV)
				hh, hw, hc = hand_hsv.shape
				hand_mean_hsv = np.mean(hand_hsv[int(hh*0.4):int(hh*0.6),int(hw*0.4):int(hw*0.6),:], axis=(0,1))

				fh, fw, fc = roi_color.shape
				face_mean = np.mean(roi_color[int(fh*0.25):int(fh*0.75),int(fw*0.25):int(fw*0.75),:], axis=(0,1))
				face_mean_rgb = np.ones((1,1,3)) * face_mean
				face_mean_hsv = cv2.cvtColor(np.array(face_mean_rgb, dtype=np.uint8), cv2.COLOR_BGR2HSV)
				face_mean_hsv = face_mean_hsv.astype('float32')
				value_diff = face_mean_hsv[0,0,2] - hand_mean_hsv[2]

				for y in range(0, hh):
					for x in range(0, hw):
						hand_hsv[y,x,0] = face_mean_hsv[0,0,0]
						hand_hsv[y,x,2] = max(0,min(hand_hsv[y,x,2] + value_diff,255))

				hand_bgr = cv2.cvtColor(hand_hsv, cv2.COLOR_HSV2BGR)
				b, g, r, a = cv2.split(hands[i])
				hand_bgra = cv2.merge((hand_bgr,a))
				hand_bgra = cv2.resize(hand_bgra, tuple(image.shape[:2]))
				hand_bgra = imutils.rotate_bound(hand_bgra, randint(0, 360))
				hand_center = (iw * (0.25 + 0.5 * random()), ih * (0.25 + 0.5 * random()))
				objectOverlay(image, hand_bgra, left_eye[0] - middle_eye[0], hand_center, labels, "background")

			# Stripe occluder
			else:

				occluded = True
				stripe_top_left = (int(iw * (0.0  + 0.5 * random())), int(ih * (0.0  + 0.5 * random())))
				stripe_size = (int(iw * (0.25 + 0.33 * random())), int(ih * (0.25 + 0.33 * random())))
				stripe_bottom_left = (stripe_size[0] + stripe_top_left[0], stripe_size[1] + stripe_top_left[1])
				stripe_color  = (255 * random(), 255 * random(), 255 * random()) 
				cv2.rectangle(image, stripe_top_left, stripe_bottom_left, stripe_color, -1)
				cv2.rectangle(labels, stripe_top_left, stripe_bottom_left, label_colors[0], -1)

		# Save output
		rndstamp = str( int(random() * 1E8) )
		rndstamp = rndstamp.zfill(8)

		if occluded:
			augmented_face_file = output_folder_faces_occluded + name + '_' + rndstamp + '.jpg'
			labels_file = output_folder_labels_occluded + name + '_' + rndstamp + '.ppm'
		else:
			augmented_face_file = output_folder_faces + name + '_' + rndstamp + '.jpg'
			labels_file = output_folder_labels + name + '_' + rndstamp + '.ppm'

		cv2.imwrite(augmented_face_file, image)
		cv2.imwrite(labels_file, labels)

		continue

	print("Processed " + face_file)

	# Show output
	# cv2.imshow(face_file, image)
	# if cv2.waitKey(0) & 0xFF == 27:
	# 	exit(0)
	# cv2.destroyWindow(face_file)	

cv2.destroyAllWindows()	