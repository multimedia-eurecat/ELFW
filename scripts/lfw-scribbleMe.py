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
super_scribbles = []
isDrawing = False
radius = 10
category = 1

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

		# Scribbles to SP elections
		super_scribbles = np.zeros(scribbles.shape)
		sp_votes = {}
		sp_areas = np.zeros(index+1)
		h, w = sp_reindex.shape

		for y in range(0, h):
			for x in range(0, w):
				
				s = sp_reindex[y, x]
				sp_areas[s] = sp_areas[s] + 1

				vote_rgb = scribbles[y,x]
				if vote_rgb.any():
					if s not in sp_votes:
						sp_votes[s] = {}

					vote_rgb = tuple(vote_rgb)
					if vote_rgb in sp_votes[s]:
						sp_votes[s][vote_rgb] = sp_votes[s][vote_rgb] + 1
					else:
						sp_votes[s][vote_rgb] = 1
					

		for s in sp_votes.keys():
			winner, votes = max(sp_votes[s].items(), key=operator.itemgetter(1))
			super_scribbles[sp_reindex == s] = np.array(winner)# (0,255,0)

	if isDrawing and (event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE):

		cv2.circle(scribbles, pointer, radius, label_colors[category], -1)

# ---------------------------------------------------------------------------------------

if len(sys.argv) != 4:
	print("Usage: $ lfw-scribbleMe <faces_folder> <superpixels_folder> <output_folder>")
	exit(0)

faces_folder 	= sys.argv[1]
sp_folder 		= sys.argv[2]
output_folder 	= sys.argv[3]

if not os.path.exists(output_folder):
	os.mkdir(output_folder)

# faces_folder = '../Datasets/lfw-deepfunneled/'
# sp_folder = '../Datasets/lfw-deepfunneled-sp/'
# output_folder = '../Datasets/lfw-deepfunneled-sp-overlay/'

for face_file in sorted(os.listdir(faces_folder)):

	if not face_file.endswith(".jpg"):
		continue

	file_name = os.path.splitext(face_file)[0]
	super_scribbles_file = os.path.join(output_folder, file_name + '.png')
	if os.path.exists(super_scribbles_file):
		continue

	face = cv2.imread(os.path.join(faces_folder, face_file))
	person_name = file_name[:-5] 
	sp_file = os.path.join(os.path.join(sp_folder, person_name), file_name + '.dat')

	if not os.path.exists( sp_file ):
		print('\033[1m' + 'Superpixels not found in ' + sp_file + '\033[0m')
		exit(0)

	print('Editing ' + '\033[1m' + file_name + '\033[0m' + "...")

	# Superpixels: watch out, SP do not have univoque numbering
	sp = np.fromfile(sp_file, dtype=int, count=-1, sep=' ')
	sp = np.array(sp, dtype=np.uint8)
	sp = np.reshape(sp, (250, -1))
	h, w = sp.shape

	# Superpixels bounds
	bounds = np.zeros(sp.shape)
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

	# Boundaries visualization
	b,g,r = cv2.split(face)
	r[bounds > 0] = r[bounds > 0] * 0.2 + 255 * 0.8;
	bounds = cv2.merge((b,g,r))

	## SP re-indexing: there could be several superpixels for each SP index label
	index = 0
	sp_reindex = np.zeros(sp.shape, dtype='uint32')
	for s in range(0,np.amax(sp)+1):
		mask = np.zeros(sp.shape, dtype='uint8')
		mask[sp == s] = 255
		_, components = cv2.connectedComponents(mask, connectivity=4)

		if np.amax(components):
			for c in range(1,np.amax(components)+1):
				index = index + 1
				sp_reindex[components == c] = index

	# Scribbles
	scribbles = np.zeros(face.shape)
	super_scribbles = scribbles.copy()
	face_canvas = face.copy()

	# Mouse events callback
	cv2.namedWindow(file_name)
	cv2.setMouseCallback(file_name, onClick)

	# Defaults
	radius = 2
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
			radius = max(radius - 2, 2)
		elif k == 32:
			if radius < 10:
				radius = 16
			else:
			 	radius = 2
		elif k == 13:
			break
		elif k == 27:
			exit(0)

		# Compositing
		alpha = 0.12
		face_canvas = face.copy()
		face_canvas[super_scribbles != 0] = face_canvas[super_scribbles != 0] * alpha + super_scribbles[super_scribbles != 0] * (1-alpha)

		alpha = 0.12
		bounds_canvas = bounds.copy()
		bounds_canvas[scribbles != 0] = bounds_canvas[scribbles != 0] * alpha + scribbles[scribbles != 0] * (1-alpha)

		alpha = 0.5
		overlay = bounds_canvas.copy()
		cv2.circle(overlay, pointer, radius, label_colors[category], -1)
		bounds_canvas = cv2.addWeighted(bounds_canvas, alpha, overlay, 1 - alpha, 0)

		vis = np.concatenate((bounds_canvas, face_canvas), axis=1)
		vis = cv2.resize(vis, (vis.shape[1] * resize, vis.shape[0] * resize), cv2.INTER_NEAREST) 

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
		info = "Save and give me more (enter)"
		cv2.putText(vis, info, (10, hstep * 3), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255)) 		
		info = "Exit (esc)"
		cv2.putText(vis, info, (10, hstep * 4), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255,255,255)) 

		cv2.imshow(file_name, vis)

	cv2.destroyWindow(file_name)

	# Save output
	cv2.imwrite(super_scribbles_file, super_scribbles)
	print("Labels saved in " + super_scribbles_file)

cv2.destroyAllWindows()