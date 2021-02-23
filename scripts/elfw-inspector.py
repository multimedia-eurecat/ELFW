# This code visualizes superpixels of the Labeled Faces in the Wild dataset.
# R. Redondo, Eurecat 2019 (c).

import numpy as np
import sys
import cv2
import os

if len(sys.argv) != 4:
	print("Usage: $ elfw-inspector <people faces folder> <superpixels folder> <output folder>")
	exit(0)

faces_folder = sys.argv[1]
sp_folder = sys.argv[2]
output_folder = sys.argv[3]

# faces_folder = '../Datasets/lfw-deepfunneled/'
# sp_folder = '../Datasets/lfw-deepfunneled-sp/'
# output_folder = '../Datasets/lfw-deepfunneled-sp-overlay/'

if not os.path.exists(output_folder):
	os.mkdir(output_folder)

for person in os.listdir(faces_folder):

	face_path = os.path.join(faces_folder, person)

	if not os.path.isdir(face_path):
		continue

	sp_path = os.path.join(sp_folder, person)

	if not os.path.isdir(sp_path):
		continue

	for face_file in os.listdir(face_path):
		
		if not face_file.endswith(".jpg"):
			continue

		print('Processing file ' + face_file)

		name = os.path.splitext(face_file)[0]
		sp_file = os.path.join(sp_path, name + '.dat')

		if not os.path.exists( sp_file ):
			print('\033[1m' + 'Superpixels not found in ' + sp_file + '\033[0m')
			continue

		# Face image
		image = cv2.imread(os.path.join(face_path, face_file))

		# Superpixels
		sp = np.fromfile(sp_file, dtype=int, count=-1, sep=' ')
		sp = np.array(sp, dtype=np.uint8)
		sp = np.reshape(sp, (250, -1))

		# Superpixels bounds
		bounds = np.zeros(sp.shape)
		h, w = bounds.shape
		for y in range(0, h):
			for x in range(0, w):
				if y > 0:
					if sp[y, x] != sp[y-1, x  ]:
						bounds[y,x] = 255;
						continue
				if y < h-1:
					if sp[y, x] != sp[y+1, x  ]:
						bounds[y,x] = 255;
						continue
				if y < h-1 and x > 0:
					if sp[y, x] != sp[y+1, x-1]:
						bounds[y,x] = 255;
						continue
				if y < h-1 and x < w-1:
					if sp[y, x] != sp[y+1, x+1]:
						bounds[y,x] = 255;
						continue
				if y > 0 and x > 0:							
					if sp[y, x] != sp[y-1, x-1]:
						bounds[y,x] = 255;
						continue
				if y > 0 and x < w-1:
					if sp[y, x] != sp[y-1, x+1]:
						bounds[y,x] = 255;
						continue
				if x > 0:
					if sp[y, x] != sp[y  , x-1]:
						bounds[y,x] = 255;
						continue
				if x < w-1:
					if sp[y, x] != sp[y  , x+1]:
						bounds[y,x] = 255;
						continue

		# Erode
		kernel = np.ones((2,2),np.uint8)
		bounds = cv2.erode(bounds, kernel, iterations = 1)

		# Visualization
		sp = cv2.cvtColor(sp, cv2.COLOR_GRAY2RGB)
		b,g,r = cv2.split(image)
		r[bounds > 0] = r[bounds > 0] * 0.2 + 255 * 0.8;
		bounds = cv2.merge((b,g,r))
		vis = np.concatenate((image, sp), axis=1)
		vis = np.concatenate((vis, bounds), axis=1)

		# Save output
		sp_overlay_file = os.path.join(output_folder, name + '.jpg')
		cv2.imwrite(sp_overlay_file, bounds)

		# Show output
		# cv2.imshow(face_file, vis)
		# if cv2.waitKey(0) & 0xFF == 27:
		# 	exit(0)
		# cv2.destroyWindow(face_file)	

cv2.destroyAllWindows()