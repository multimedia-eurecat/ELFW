# This code calculates appearance frequencies and area occupation of the ELFW classes.
# R. Redondo (c) Eurecat 2019

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = '../Datasets/lfw-bagsoffaces/elfw/elfw_01_basic/labels'

if not os.path.exists(path):
    print('Input path not found ' + path)
    exit(-1)

label_colors = [
    (0, 0, 0),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255)]
    # (255, 255, 0)]

label_names = [
    "bkgnd",
    "skin",
    "hair",
    "beard",
    "sunglasses",
    "wearable"]
    # "mask"]

def set_fontsize(type):

    if type == 'normal':
        size = 9
    elif type == 'small':
        size = 7

    font = {'family': 'normal', 'weight': 'normal', 'size': size}
    plt.rc('font', **font)

def rotate_ticks(axis, degrees):
    for tick in axis.get_xticklabels():
        tick.set_rotation(degrees)

def format_plot(axis, title, mean, std=None):

    set_fontsize('normal')
    axis.set_title(title)
    axis.yaxis.grid(b=True, linestyle='--')
    rotate_ticks(axis, 25)
    set_fontsize('small')

    for i, m in enumerate(mean):
        label = '{:0.2f}'.format(m)
        label_size = len(label)
        y = m
        if std is not None:
            s = std[i]
            label_std = '(±' + '{:0.3f})'.format(s)
            label_size = len(label_std)
            label = '    ' + label + '\n' + label_std
            y += s + 4E-2

        x = i - label_size * 6E-2
        y += 1E-2
        axis.text(x, y, label, color='black')

    set_fontsize('normal')

# --------------------------------------------------------------------------------
# Calculate class contributions

files = os.listdir(path)
class_contributions = [ [] for l in range(len(label_names)) ]
num_extended_faces = 0

print("This will take a while...")

for file in files:

    file_path = os.path.join(path, file)
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

        if pixels:
            class_contributions[c].append(pixels)
            if c > 2:
                extended = True

    if extended:
        num_extended_faces += 1

# Plot statistics

fig_size = 3
fig, axs = plt.subplots(1, 2, figsize=(2 * fig_size, fig_size), sharey=True)
label_colors = np.array(label_colors) / 255
# set_fontsize('normal')

class_frequencies = np.array([len(c) for c in class_contributions]) / float(len(files))
axs[0].bar(list(label_names), list(class_frequencies), color=label_colors, alpha=0.8)
format_plot(axs[0], 'Normalized Appearance Frequency', class_frequencies)

size = float( image_size[0] * image_size[1] )
class_mean = np.array([np.mean(np.array(c)) for c in class_contributions]) / size
class_std = np.array([np.std(np.array(c)) for c in class_contributions]) / size

axs[1].bar(list(label_names), list(class_mean), yerr=list(class_std), color=label_colors, ecolor='black', capsize=5, alpha=0.8)
format_plot(axs[1], 'Normalized Area Occupation', class_mean, class_std)

print('Number of faces with at least 1 extended category: %d' % num_extended_faces)

plt.tight_layout()
# plt.savefig('myplot.eps')
print("Done.")
plt.show()