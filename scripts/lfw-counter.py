# This code counts face names and augmentation usages.
# R. Redondo (c) Eurecat 2019

import os
import numpy as np
import matplotlib.pyplot as plt

path = '../Datasets/lfw-bagsoffaces/elfw_handoverfaces'

if not os.path.exists(path):
    print('Input path not found ' + path)
    exit(-1)


def update_dict(dictionary, key):

    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1

# Counter

face_counter = dict(hof=dict(),h2f_web=dict())
face_counter_total = dict()
hand_counter_total = dict()

files = os.listdir(path)

for file in files:

    filename = os.path.basename(file)
    name, ext = os.path.splitext(filename)

    if ext != '.jpg':
        continue

    name_split = name.split('-')

    item = name_split[-1]
    aug_dataset = name_split[-2]
    picture = ''.join(name_split[:-2])

    # print('%s %s %s' % (picture, aug_dataset, item))

    update_dict(face_counter[aug_dataset], picture)
    update_dict(face_counter_total, picture)
    update_dict(hand_counter_total, aug_dataset+item)

# Statistics
font = {'family':'normal', 'weight':'normal', 'size':13}
plt.rc('font', **font)
fig_size = 3
subplots = len(face_counter) + 1
fig, axs = plt.subplots(1, subplots, figsize=(subplots * fig_size, fig_size), sharey=True)
titles = ['HandOverFace', 'Hand2Face']
usage_histogram_total = dict()

for idx, d in enumerate(face_counter.keys()):

    print('Total used faces with %s: %d' % (str(d), len(face_counter[d])))

    usage_histogram = dict()

    for f in face_counter[d].keys():

        usages = face_counter[d][f]

        update_dict(usage_histogram, usages)
        update_dict(usage_histogram_total, usages)

    print('Usage histogram with %s:' % str(d))
    print([(u,usage_histogram[u]) for u in sorted(usage_histogram.keys())])

    # Plot
    axs[idx].bar(list(usage_histogram.keys()), list(usage_histogram.values()))
    axs[idx].set_title(titles[idx])
    axs[idx].set_xticks(np.arange(1, np.max(np.array(list(usage_histogram.keys())))+1, step=1))

print('Total used faces: %d' % len(face_counter_total))
print('Total used hands: %d' % len(hand_counter_total))

axs[0].set_ylabel('№ hand-augmented faces')
axs[-1].bar(list(usage_histogram_total.keys()), list(usage_histogram_total.values()))
axs[-1].set_xticks(np.arange(1, np.max(np.array(list(usage_histogram_total.keys())))+1, step=1))
axs[-1].set_title('All')

for a in axs:
    a.set_xlabel('№ different hands')


plt.show()