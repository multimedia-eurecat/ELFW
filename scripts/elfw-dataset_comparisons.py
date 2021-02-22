import numpy as np
import sys
import cv2
import os

def get_masks(img, old):

	b = img[:, :, 0]
	g = img[:, :, 1]
	r = img[:, :, 2]	

	# the background mask of the original labels
	if old:
		background_mask = (r == 0) & (g == 0) & (b == 255)
	else:
		background_mask = (r == 0) & (g == 0) & (b == 0)
		
	skin_mask       = (r == 0)   & (g == 255) & (b == 0)
	hair_mask       = (r == 255) & (g == 0)   & (b == 0)

	return background_mask, skin_mask, hair_mask

base_path = "/media/jaume.gibert/Data/elfw"
images           = os.path.join(base_path, "elfw_01_basic/faces")
labels_1         = os.path.join(base_path, "parts_lfw_funneled_gt_images")
labels_2         = os.path.join(base_path, "elfw_01_basic/labels")
output           = os.path.join(base_path, "comparisons")
out_improvements = os.path.join(output, "relabelled")
out_new          = os.path.join(output, "new_labels")

if not os.path.exists(output):
	os.mkdir(output)
if not os.path.exists(out_improvements):
	os.mkdir(out_improvements)
if not os.path.exists(out_new):
	os.mkdir(out_new)

for image in os.listdir(images):

	base_name = os.path.splitext(image)[0]
	print(base_name)

	# Face image
	img_file   = os.path.join(images, image)
	lab_1_file = os.path.join(labels_1, base_name + '.ppm')
	lab_2_file = os.path.join(labels_2, base_name + '.png')

	# If all three exist, it means that we either didn't touch the labels or we improved it
	# Save into improvements 
	# TODO: check equal old and new labels and only save the improved ones
	if os.path.exists(img_file) and os.path.exists(lab_1_file) and os.path.exists(lab_2_file):
		img   = cv2.imread(img_file)
		lab_1 = cv2.imread(lab_1_file)
		lab_2 = cv2.imread(lab_2_file)

		# I take the original labels (lab_1), convert the blue (background into black)
		# and check if it's the same as in the new label image (lab_2)
		# if so, I don't want it, I want the other cases.
		b1, s1, h1 = get_masks(lab_1, old=True)
		b2, s2, h2 = get_masks(lab_2, old=False)

		if not (np.array_equal(b1, b2) and np.array_equal(s1, s2) and np.array_equal(h1, h2)):
			result = np.vstack((img, np.vstack((lab_2, lab_1))))
			cv2.imwrite(os.path.join(out_improvements, base_name+".png"), result)

	# if only the image and our labels exist, it means the image-label is new and didn't exist in LFW
	elif os.path.exists(img_file) and not os.path.exists(lab_1_file) and os.path.exists(lab_2_file):

		img   = cv2.imread(img_file)
		lab_2 = cv2.imread(lab_2_file)
		result = np.vstack((img, lab_2))
		cv2.imwrite(os.path.join(out_new, base_name+".png"), result)

	else:
		continue
	

	

