import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
# from matplotlib.ticker import MaxNLocator
# Rafael Redondo (c) Eurecat 2020

colors = np.array([
	[[247, 197, 188],[242, 151, 136],[242, 120, 99],[237, 85, 59]],
	[[255, 242, 196], [247, 232, 176], [250, 225, 135], [246, 213, 93]],
	[[180, 224, 220],[123, 209, 201],[83, 194, 183],[60, 174, 163]],
	[[159, 189, 214],[118, 160, 194],[72, 125, 168],[32, 99, 155]]
	]) / 255.0

class_colors = np.array([
	[[184, 184, 184],[125, 125, 125],[71,   71, 71], [000, 000, 000]],
	[[196, 255, 196],[178, 255, 178],[126, 255, 126],[000, 255, 000]],
	[[252, 189, 189],[255, 133, 133],[255,  77, 77], [255, 000, 000]],
	[[207, 255, 255],[176, 255, 245],[144, 255, 255],[000, 255, 255]],
	[[212, 212, 255],[149, 149, 255],[94,   94, 255],[000, 000, 255]],
	[[255, 209, 255],[255, 156, 255],[255, 101, 255],[255, 000, 255]]
	]) / 255.0


def draw(row, axis, data, method, labels, colors, xlim, hsep, radius):

	augment = data.shape[0]

	labels = list(labels)
	num_elements = len(labels)	
	# labels.insert(0,'')

	titles = ['Sunglasses Augmentation', 'Hands Augmentation', 'Sunglasses+Hands Aug.']

	for c in range(len(axis[row])):

		axis[row,c].set_aspect(1)
		axis[row,c].set_xlim(-1, xlim)
		# axis[row,c].set_ylim(-0.4, num_elements*0.5-0.1)
		axis[row, c].set_ylim(-0.4, num_elements * hsep - 0.1)

		axis[row, c].set_yticklabels(labels)
		axis[row, c].set_yticks(np.arange(num_elements) * hsep)
		# axis[row, c].yaxis.set_major_locator(MaxNLocator(integer=True))
		if c == 0:
			axis[row, c].set_ylabel(method, fontsize=fontsizeLarge)
		else:
			axis[row, c].get_yaxis().set_visible(False)
		if row == 0:
			axis[row, c].set_title(titles[c], fontsize=fontsizeMedium)

		if row == axis.shape[0] - 1:
			axis[row, c].set_xlabel('Gain', fontsize=fontsizeMedium)
		else:
			axis[row, c].get_xaxis().set_visible(False)

		axis[row, c].add_patch(Polygon([[-1, -1], [0, -1], [0, 10], [-1, 10]], closed=True, fill=True, facecolor=[0.92,0.9,0.9]))

		for m in range(num_elements):
			for s in reversed(range(augment)):

				sigma = s / float(augment-1)
				r = math.sqrt(1 + sigma) * radius
				t = m + c * num_elements
				circle = plt.Circle((data[s,t] - data[0,t], m * hsep), r, color=colors[m,augment-s-1], edgecolor=None)
				axis[row,c].add_artist(circle)

# ----------------------------------------------------------------------------------------------


fcn = np.array([
[94.86221,	89.94708,	78.54365,	90.62491,		94.86221,	89.94708,	78.54365,	90.62491,		94.86221,	89.94708,	78.54365,	90.62491],
[94.87768,	89.34935,	78.5738,	90.65072,		94.91543,	90.04007,	78.89132,	90.71592,		94.87198,	90.01351,	79.12107,	90.64796],
[94.82212,	89.07311,	78.57936,	90.55048,		95.01015,	89.30039,	79.2808,	90.85555,		94.92132,	89.81723,	79.51949,	90.71459],
[94.91106,	88.96342,	79.1459,	90.67938,		94.94046,	89.36422,	79.07776,	90.75119,		94.9023,	90.05563,	79.41762,	90.69509]
])

fcn_classes = np.array([
[94.75722,	86.38252,	71.85863,	61.34205,	72.44731,	84.47418,		94.75722,	86.38252,	71.85863,	61.34205,	72.44731,	84.47418,	94.75722,	86.38252,	71.85863,	61.34205,	72.44731,	84.47418],
[94.74529,	86.52661,	71.92953,	60.08587,	73.81507,	84.34044,		94.78213,	86.77009,	71.78261,	61.24385,	74.51008,	84.25919,	94.74461,	86.72296,	71.41357,	63.01671,	74.86192,	83.96667],
[94.70561,	86.4855,	71.23186,	60.26997,	74.76319,	84.02004,		94.91729,	86.89263,	72.04419,	62.55301,	75.33231,	83.94535,	94.789,		86.70316,	71.52878,	63.19029,	75.79578,	85.10994],
[94.74795,	86.68139,	71.68878,	62.33461,	74.78171,	84.64096,		94.79637,	86.70211,	71.9773,	61.76077,	73.98104,	85.249,		94.72738,	86.72993,	71.767,		62.76649,	75.47534,	85.03957]

])

deeplab = np.array([
[94.6848,	89.71417,	77.94909,	90.37054,		94.6848,	89.71417,	77.94909,	90.37054,		94.6848,	89.71417,	77.94909,	90.37054],
[94.78537,	89.59541,	78.56921,	90.51187,		94.86725,	89.7494,	78.62243,	90.63049,		94.81017,	89.91979,	78.41131,	90.55676],
[94.82899,	90.35099,	79.05202,	90.57593,		94.86047,	90.18027,	78.72145,	90.63608,		94.90303,	90.12334,	79.14438,	90.70572],
[94.89329,	90.06537,	79.38813,	90.67735,		94.9435,	90.07861,	78.87746,	90.75484,		94.89794,	90.37945,	79.37854,	90.70328]
])

deeplab_classes = np.array([
[94.51156,	86.31067,	71.33108,	60.57315,	71.34837,	83.61973,		94.51156,	86.31067,	71.33108,	60.57315,	71.34837,	83.61973,		94.51156,	86.31067,	71.33108,	60.57315,	71.34837,	83.61973],
[94.63511,	86.41761,	71.65853,	61.14348,	74.20494,	83.35558,		94.77132,	86.44033,	71.84831,	62.07319,	72.95129,	83.65015,		94.66345,	86.50839,	71.66187,	59.54201,	74.23029,	83.86187],
[94.66677,	86.3675,	71.90078,	61.5994,	75.30071,	84.47695,		94.73285,	86.59962,	71.78432,	62.6076,	72.75062,	83.8537,		94.75528,	86.66354,	71.85329,	61.44883,	74.97043,	85.17492],
[94.75281,	86.6263,	71.72533,	63.26911,	75.43251,	84.52274,		94.81422,	86.6394,	72.28983,	61.98376,	73.2231,	84.31447,		94.72211,	86.83926,	71.83235,	63.31498,	75.25782,	84.30474]
])


fontsizeSmall = 12
fontsizeMedium = 16
fontsizeLarge = 18

font = {'family':'normal', 'weight':'normal', 'size': fontsizeSmall}
plt.rc('font', **font)

metrics_fig, metrics_axis = plt.subplots(2, 3, sharey=True, sharex=True)
draw(0, metrics_axis, fcn, 		'FCN', 			('Pixel Acc.', 'Mean Acc.', 'Mean IU', 'Freq.W. IU'), colors, 1.8, 0.4, 0.1)
draw(1, metrics_axis, deeplab,	'DeepLabV3', 	('Pixel Acc.', 'Mean Acc.', 'Mean IU', 'Freq.W. IU'), colors, 1.8, 0.4, 0.1)

# class_fig, class_axis = plt.subplots(2, 3, sharey=True)
# draw(0, class_axis, fcn_classes, 	'FCN',		('Bkgnd', 'Skin', 'Hair', 'Beard', 'Snglss', 'Wear'), class_colors, 4.5, 0.5, 0.15)
# draw(1, class_axis, deeplab_classes,'DeepLabV3',('Bkgnd', 'Skin', 'Hair', 'Beard', 'Snglss', 'Wear'), class_colors, 4.5, 0.5, 0.15)

plt.show()