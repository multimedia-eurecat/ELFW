from PIL import Image
import numpy as np
import os
#from import check_mkdir, bcolors
import sys


target_size = 256

# I/O
augmentation_ratios = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
epochs              = range(0, 201, 10)
images_folder       = "/media/jaume.gibert/Data/elfw/elfw_01_basic/faces"
labels_folder       = "/media/jaume.gibert/Data/elfw/elfw_01_basic/labels"
predictions_folder  = "/home/jaume.gibert/Code/facesinthewild/predictions/Background_Vs_Sunglasses/lr1e-4"

# Input From File
f = open("/media/jaume.gibert/Data/elfw/elfw_01_basic/elfw_set_00.txt", "r")
names = []
for line in f:
    # for some reason it's also loading the \n at the end of each line
    if line[-1:]=='\n':
        names.append(line[:-1])
    else:
        names.append(line)

def main():

	for image_name in names:

		# Read the image (resize it) and labels
		image_name_path = os.path.join(images_folder, image_name + ".jpg")
		image = Image.open(image_name_path).convert("RGB")
		image = image.resize((target_size, target_size), Image.BILINEAR)

		label_name_path = os.path.join(labels_folder, image_name + ".png")
		label = Image.open(label_name_path).convert("RGB")
		label = label.resize((target_size, target_size), Image.BILINEAR)

		A = np.concatenate((np.array(image), np.array(label)), axis=1)
		B = np.zeros((target_size, 2*target_size, 3))

		for rho in augmentation_ratios:

			path = os.path.join(predictions_folder, "Aug_ratio_"+str(rho))
			out = A if rho == 0 else B
			for ep in epochs:

				prediction_path = os.path.join(path, image_name+"_gcn-epoch_"+str(ep).zfill(4)+".png")
				prediction      = Image.open(prediction_path).convert("RGB")
				prediction      = prediction.resize((target_size, target_size), Image.BILINEAR)
				out      		= np.concatenate((out, prediction), axis=1)
			
			composition = out if rho==0 else np.concatenate((composition, out), axis=0)

		composition = Image.fromarray(composition.astype('uint8'))
		output_file = os.path.join(predictions_folder, 'compositions', image_name+".jpg")
		composition.save(output_file)


if __name__ == '__main__':
    main()
