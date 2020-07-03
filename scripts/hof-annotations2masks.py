# Converts HandOverFace annotations to binary masks
# Rafael Redondo (c) Eurecat 2019

# Erratas in the original dataset:
# 10.jpg does not match 10.png neither 10.xml
# 216.jpg and 221.jpg are actually a GIFs in the original size folder
# 225.jpg contains a body a part from hands

import os
import cv2
import numpy as np

annotations_path = '../Datasets/Hand_datasets/hand_over_face/annotations'
handfaces_path = '../Datasets/Hand_datasets/hand_over_face/images_original_size'
output_path = '../Datasets/Hand_datasets/HOF_highres_mask'

annotations = os.listdir(annotations_path)
handfaces = os.listdir(handfaces_path)

import xml.etree.ElementTree as ET

# Run over all annotation files
for _, file in enumerate(annotations):

	basename, extension = os.path.splitext(file)
	if extension != '.xml':
		continue

	print('Processing file \033[1m%s\033[0m' % file)

	xml_file = os.path.join(annotations_path, file)
	parsed_annotation = ET.parse(xml_file)
	xml_root = parsed_annotation.getroot()

	handface_file = os.path.join(handfaces_path, basename + '.jpg')
	handface = cv2.imread(handface_file)
	nrows , ncols, _ = handface.shape
	mask = np.zeros((nrows, ncols))

	# Run over all objects and polygons
	for elem in xml_root:
		if elem.tag != 'object':
			continue

		for subelem in elem:
			if subelem.tag != 'polygon':
				continue

			vertices = []
			for subsubelem in subelem:
				if subsubelem.tag == 'pt':
					x = int(subsubelem[0].text)
					y = int(subsubelem[1].text)
					vertices.append((x,y))

			if len(vertices):
				print('Found %d vertices' % len(vertices))
				cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), 255)
			else:
				print('Wop! no vertices found in ' + xml_file)
				continue

	mask_file = os.path.join(output_path, basename + '.png')
	cv2.imwrite(mask_file, mask)
	# test_file = os.path.join(output_path, 'test-' + basename + '.png')
	# mmm = cv2.merge((mask, mask, mask))
	# cv2.imwrite(test_file, handface * mmm / 255)
	# cv2.imshow('mask', mask)
	# cv2.waitKey()
