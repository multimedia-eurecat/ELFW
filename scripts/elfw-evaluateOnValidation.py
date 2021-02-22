from PIL import Image
import numpy as np
import os
#from import check_mkdir, bcolors
import sys
import matplotlib.pyplot as plt
import fnmatch

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

# I/O
images_folder      = "/media/jaume.gibert/Data/elfw/elfw_01_basic/faces"
labels_folder      = "/media/jaume.gibert/Data/elfw/elfw_01_basic/labels"
predictions_folder = "/media/jaume.gibert/Data/elfw/predictions"
output_folder      = "/media/jaume.gibert/Data/elfw/eval_curves"

# Input From File
f = open("/media/jaume.gibert/Data/elfw/elfw_01_basic/elfw_set_00.txt", "r")
names = []
for line in f:
    # for some reason it's also loading the \n at the end of each line
    if line[-1:]=='\n':
        names.append(line[:-1])
    else:
        names.append(line)

target_size  = 256

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

num_classes = len(label_colors)

def ToELFWLabels(data):

	data = np.array(data)

	r = data[:, :, 0]
	g = data[:, :, 1]
	b = data[:, :, 2]

	output = np.zeros((data.shape[0], data.shape[1]))
	for c in range(0,len(label_colors)):
		color_mask = (r == label_colors[c][0]) & (g == label_colors[c][1]) & (b == label_colors[c][2])
		output[color_mask] = c

	return output		


def main():

	# list folders in checkpoints folder
	for config in os.listdir(predictions_folder):
		
		print(config)
		path = os.path.join(predictions_folder, config)		
		
		# we need to find out how many epochs have been computed for this configuration
		# this should not be like this, since we expect to have the same number of epochs for
		# each configuration, but the experiment crashed and we want to find out how many epochs
		# we have
		# HOW: take the first image of the validation set, and see which epochs are there
		# for each image, the epochs should be exactly the same
		image_name = names[0]
		epochs = []
		for file in os.listdir(path):
			if fnmatch.fnmatch(file, "*"+image_name+"*"):
				epochs.append(int(file[-8:-4]))
		epochs.sort()
		
		#epochs = [0, 10]

		accuracy_curves  = np.zeros((num_classes, len(epochs)))
		precision_curves = np.zeros((num_classes, len(epochs)))
		recall_curves    = np.zeros((num_classes, len(epochs)))
		fscore_curves    = np.zeros((num_classes, len(epochs)))

		
		for idx_ep, ep in enumerate(epochs):

			#print("   > Epoch " + str(ep))

			count = np.zeros((num_classes, 1))

			for image_name in names:

				# Read the labels
				label_name_path = os.path.join(labels_folder, image_name + ".png")
				label = Image.open(label_name_path).convert("RGB")
				label = label.resize((target_size, target_size), Image.NEAREST)
				label = ToELFWLabels(label)	

				prediction_path = os.path.join(path, image_name+"_fcn-epoch_"+str(ep).zfill(4)+".png")
				prediction      = Image.open(prediction_path).convert("RGB")
				prediction      = prediction.resize((target_size, target_size), Image.NEAREST) # this should not be necessary since predictions are already in target_size size
				prediction      = ToELFWLabels(prediction)

				for c in range(0, num_classes):
					
					A = (prediction == c)
					B = (label      == c)
					C = (prediction != c)
					D = (label      != c)

					# True Positive (TP): we predict a label of class cl (positive), and the true label is cl.
					TP_mask = np.logical_and(A, B)
					TP = np.sum(TP_mask)
					# True Negative (TN): we predict a label that it's not 0 (negative), and the true label is not cl.
					TN_mask = np.logical_and(C, D)
					TN = np.sum(TN_mask)
					# False Positive (FP): we predict a label of class cl (positive), but the true label is not cl.
					FP_mask = np.logical_and(A, D)
					FP = np.sum(FP_mask)
					# False Negative (FN): we predict a label that it's not cl (negative), but the true label is cl.
					FN_mask = np.logical_and(C, B)
					FN = np.sum(FN_mask)

					if np.sum(np.logical_or(A, B)):

						Accuracy  = (TP+TN) / (TP+TN+FP+FN+1E-8)
						Precision = (TP) / (TP+FP+1E-8)
						Recall    = (TP) / (TP+FN+1E-8)
						F_Score   = (2*Precision*Recall) / (Precision+Recall+1E-8)

						accuracy_curves [c, idx_ep] += Accuracy
						precision_curves[c, idx_ep] += Precision
						recall_curves   [c, idx_ep] += Recall
						fscore_curves   [c, idx_ep] += F_Score

						count[c] += 1	

			for c in range(num_classes):
				if count[c]:
					accuracy_curves[c, idx_ep]  /= float(count[c])
					precision_curves[c, idx_ep] /= float(count[c])
					recall_curves[c, idx_ep]    /= float(count[c])
					fscore_curves[c, idx_ep]    /= float(count[c])                 
		
		
		output_path = os.path.join(output_folder, config)
		if not os.path.exists(output_path):
			os.mkdir(output_path)
		
		# Accuracy
		fig = plt.figure()
		for c in range(num_classes):
			plt.plot(epochs, accuracy_curves[c, :], label=label_names[c])
		plt.title('Accuracy')
		plt.legend()
		plt.grid(axis='y')
		plt.savefig(os.path.join(output_path, '00-Accuracy.png'))

		# Precision
		fig = plt.figure()
		for c in range(num_classes):
			plt.plot(epochs, precision_curves[c, :], label=label_names[c])
		plt.title('Precision')
		plt.legend()	
		plt.grid(axis='y')
		plt.savefig(os.path.join(output_path, '01-Precision.png'))

		# Recall
		fig = plt.figure()
		for c in range(num_classes):
			plt.plot(epochs, recall_curves[c, :], label=label_names[c])
		plt.title('Recall')
		plt.legend()	
		plt.grid(axis='y')
		plt.savefig(os.path.join(output_path, '02-Recall.png'))

		# F-Score
		fig = plt.figure()
		for c in range(num_classes):
			plt.plot(epochs, fscore_curves[c, :], label=label_names[c])
		plt.title('F-Score')
		plt.legend()	
		plt.grid(axis='y')
		plt.savefig(os.path.join(output_path, '03-F_Score.png'))


		# Display max values of f-score
		print("")
		for c in range(num_classes-1):
			curve = fscore_curves[c,:]
			idx = np.argmax(curve)
			if c==1 or c==2:
				print(bcolors.BOLD + label_names[c] + bcolors.ENDC + "\t\tMax F-Score: "+ bcolors.BLUE+str(curve[idx])+ bcolors.ENDC + " at epoch " + str(epochs[idx]))
			else:
				print(bcolors.BOLD + label_names[c] + bcolors.ENDC + "\tMax F-Score: "+ bcolors.BLUE+str(curve[idx])+ bcolors.ENDC + " at epoch " + str(epochs[idx]))
		print("")


if __name__ == '__main__':
    main()
