# From the original SegNet paper:
#
# We use median frequency balancing [13] where the weight assigned to a class in the loss function
# is the ratio of the median of class frequencies computed on the entire training set divided by the class frequency.

# If we go to the reference the SegNet authors refer to:

# we weight each pixel by a_c = median_freq / freq(c) where freq(c) is the number of pixels of class c divided by the
# total number of pixels in images where c is present, and median_freq is the median of these frequencies.

import os
import numpy as np
from PIL import Image
import cv2
import sys 

def ToELFWLabel(data, label_colors):

    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]

    output = np.zeros((data.shape[0], data.shape[1]))
    for c in range(0,len(label_colors)):
        color_mask = (r == label_colors[c][0]) & (g == label_colors[c][1]) & (b == label_colors[c][2])
        output[color_mask] = c

    return output

if len(sys.argv) != 2:
    print "Usage: $ computeClassWeights <labels folder>"
    exit(0)

label_colors = [
    (0, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (255, 255, 0)]

label_names = [
    "background",
    "skin",
    "hair",
    "beard-mustache",
    "sunglasses",
    "wearable",
    "mouth-mask"]

src_folder = sys.argv[1]
n = len(label_names)
px_frequencies = np.zeros(n)
im_frequencies = np.zeros(n)

print 'Please wait. Processing dataset...'

for f in os.listdir(src_folder):
    file_name = os.path.join(src_folder, f)
    image = Image.open(file_name).convert("RGB")
    img = np.array(image)
    img = ToELFWLabel(img, label_colors)
    for l in range(0, n):
        px = np.sum(img==l)
        # label counts if it is present in the image
        if px > 0:
            px_frequencies[l] += px
            im_frequencies[l] += img.size

# Mask for indices of appearing classes in the train set
m   = (px_frequencies>0)
idx = np.where(m)

frequencies            = np.divide(px_frequencies[m],  im_frequencies[m])
pos_median_frequencies = np.divide(np.median(frequencies), frequencies)

median_frequencies = np.zeros(n)
for l in range(0,len(pos_median_frequencies)):
    median_frequencies[idx[0][l]] = pos_median_frequencies[l]

for l in range(0,len(median_frequencies)):
    print "Weight %f for class %s" % (median_frequencies[l], label_names[l])

