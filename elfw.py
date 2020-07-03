import os
import os.path as osp
import sys
import numpy as np
import random
from PIL import Image
import collections
import torch
from torch.utils import data
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from transform import Scale
from utils import bcolors

# This file contains the ELFW Dataset.
#
# Important variables:
#
#   dataset_path                    -> path to the dataset containing the ELFW faces and labels.
#   synthetic_paths                 -> a dictionary of name-paths pairs containing synthetic datasets (faces and labels).
#
# Organize your datasets as follows:
#
#   dataset/faces                   -> a folder containing all input faces.
#   dataset/labels                  -> a folder containing all labels paired to the faces.
#
# Also, populate the file named as 'dataset_path/elfw_set_{valset}.txt' a list of names to be used for validation, and so excluded from training.
#
# Overfitting with a single image:
#
#   1. Place the image you want to do overfitting with in the 'faces' folder.
#   2. Do the same with its label image in the 'labels' folder.
#   3. Make a copy of them, rename them and place the in the same 'faces' and 'labels' folder respectively.
#   4. Make sure one of the two faces file names is listed in 'elfw_set_{valset}.txt'.
#
# Rafael Redondo & Jaume Gibert - Eurecat (c) 2019

# Cluster
dataset_path    = "/media/ssd2/elfw/elfw_01_basic"
synthetic_paths = [{"name": "Sunglasses",
                   "path": "/media/ssd2/elfw/elfw_AugmentedGlasses"},
                  {"name": "Hands",
                   "path": "/media/ssd2/elfw/elfw_AugmentedHands"},
                  {"name": "Masks",
                   "path": "/media/ssd2/elfw/elfw_AugmentedMasks"}]

# # Local
# dataset_path    = "/media/jaume.gibert/Data/elfw/elfw_01_basic"
# synthetic_paths = [{"name": "Sunglasses",
#                    "path": "/media/jaume.gibert/Data/elfw/elfw_AugmentedGlasses"},
#                   {"name": "Hands",
#                    "path": "/media/jaume.gibert/Data/elfw/elfw_AugmentedHands"},
#                   {"name": "Masks",
#                    "path": "/media/jaume.gibert/Data/elfw/elfw_AugmentedMasks"}]

# # Local
# dataset_path    = "/media/jaume.gibert/Data/elfw/debug/train"
# synthetic_paths = [{"name": "Sunglasses",
#                    "path": "/media/jaume.gibert/Data/elfw/debug/aug0"},
#                   {"name": "Hands",
#                    "path": "/media/jaume.gibert/Data/elfw/debug/aug1"},
#                   {"name": "Masks",
#                    "path": "/media/jaume.gibert/Data/elfw/debug/aug2"}]

class ELFWDataSet(data.Dataset):

    def __init__(self, 
                split='train',                  # Either train or validation
                valset=0,                       # Specifies the number partition for validation images
                random_transform=False,         # Boolean for random data augmentation
                synth_augmen_types=None,        # List of indices for folders of images that will be used as class augmentation
                synth_augmen_ratio=0,           # Percentage (wrt the training images) from augmentation folder that will be included in the train set
                compute_class_weights=False,    # If True it computes class weights for the whole dataset (original + augmented)
                excluded_classes=None):         # List of integers specifying the classes that are not gonna be used

        # Dataset Labels: number of categories, names, and associated colors (the very first to be computed)
        self.update_classes(excluded=excluded_classes)

        self.root                  = dataset_path
        self.split                 = split 
        self.valset                = valset 
        self.files                 = collections.defaultdict(list) # pairs of image+label names separated in train and validation
        self.random_transform      = random_transform 
        self.median_frequencies    = np.ones(self.num_classes)

        # Image and label transformations
        self.target_size = 256
        self.img_transform = Compose([
            Scale((self.target_size, self.target_size), Image.BILINEAR),
            ToTensor(),
            Normalize([.485, .456, .406], [.229, .224, .225])  # Useful when using pre-trained nets
            ])

        self.source_size = 250
        self.label_transform = Compose([
            ToELFWLabel(self.label_colors, self.source_size),
            Scale((self.target_size, self.target_size), Image.NEAREST),
            ])

        # Populating the dataset
        val_set_name  = "elfw_set"
        faces_folder  = "faces"
        labels_folder = "labels"

        val_set   = osp.join(self.root, val_set_name + "_%s.txt" % str(self.valset).zfill(2))
        val_file  = open(val_set,"r")
        val_names = [osp.splitext(file.strip())[0] for file in val_file]
        val_file.close()

        face_files = []

        if self.split == 'train':
            face_files = os.listdir(osp.join(self.root, faces_folder))
        elif self.split == 'validation':
            face_files = val_names
        elif self.split == 'test':
            return
        else:
            print("Error: undefined split type!")
            exit(1)

        print(bcolors.YELLOW + "Loading ELFW split \'%s\' from %s" % (split, self.root) + bcolors.ENDC)

        for filename in face_files:
            name = osp.splitext(filename)[0]

            if self.split == 'train':
                if name in val_names:   # Skip validation images from training
                    continue

            img_file   = osp.join(self.root, osp.join(faces_folder,  "%s.jpg" % name))
            label_file = osp.join(self.root, osp.join(labels_folder, "%s.png" % name))

            if not osp.exists(label_file):
                print(bcolors.BOLD + 'Labels not found in ' + label_file + bcolors.ENDC)
                continue

            self.files[self.split].append({
                "img": img_file,
                "label": label_file
                })

        # Define the augmentation folders that to be used
        if self.split == 'train':
            self.augmentation_folders(synth_augmen_types)

        # Add images from the synth_augmen_folder (if requested)
        if self.split == 'train' and self.synth_augmen_folders and synth_augmen_ratio > 0:

            n_train_images     = len(self.files[self.split])
            n_aug_images_total = int(synth_augmen_ratio * n_train_images)
            n_aug_images_part  = int(n_aug_images_total / len(self.synth_augmen_folders))

            for sf in self.synth_augmen_folders:

                synth_augmen_folder = sf['path']
                synth_aug_files = os.listdir(osp.join(synth_augmen_folder, 'faces'))
                print((bcolors.BLUE + "Synthetic augmentation: %d out of %d images for %s at %s" + bcolors.ENDC) % \
                                    (n_aug_images_part, len(synth_aug_files), sf['name'], synth_augmen_folder))

                # Shuffle all augmentation images and keep adding them until we have as much as n_aug_images
                random.shuffle(synth_aug_files)

                c = 0
                for aug_filename in synth_aug_files:

                    # remove the extension
                    name = osp.splitext(aug_filename)[0]

                    # Check if this image belongs to the validation set also, this is, if it is an image from 
                    # the validation set that has been augmented. In this case, we discard it:
                    # All augmented images are composed of the original name of the person
                    # and an augmentatio ID for the different assets. this ID always starts with '_elfw'
                    # so it's something like name_surname_0003_elfw-sunglasses-12
                    # we need to know if the name_surname_0003 part is in the val_names list
                    if name[:name.find("_elfw")] in val_names:
                        continue

                    # Get image and labels names
                    img_file   = osp.join(synth_augmen_folder, 'faces',  "%s.jpg" % name)
                    label_file = osp.join(synth_augmen_folder, 'labels', "%s.png" % name)

                    # Check existence of label file - for security
                    if not osp.exists(label_file):
                        print(bcolors.RED + 'Labels not found in ' + label_file + bcolors.ENDC)
                        continue

                    # Add pair into the training list
                    self.files[self.split].append({
                        "img": img_file,
                        "label": label_file
                    })

                    # Check how many augmentation images have been used so far. If max is reached, then break
                    c += 1
                    if (c == n_aug_images_part):
                        break

            # Shuffle images so they get mixed across different synthetic augmentation folders
            random.shuffle(self.files[self.split])                

        if compute_class_weights and self.split == 'train':
            print(bcolors.GREEN + "Computing class balancing weights..." + bcolors.ENDC)
            self.__compute_class_balance_weights__()

        print(("Loaded " + bcolors.BOLD + "%s" + bcolors.ENDC + " split with " + bcolors.BOLD + "%d" + bcolors.ENDC + " items")
            % (self.split, len(self.files[self.split])) )

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        image_file = datafiles["img"]
        label_file = datafiles["label"]

        image = Image.open(image_file).convert('RGB')
        label = Image.open(label_file).convert("RGB")

        if self.random_transform:
            # Flip
            if random.random() > 0.5:
                image = F.hflip(image)
                label = F.hflip(label)
            # Shift
            shift = [0, 0]
            if random.random() > 0.5:
                shift[0] = (random.random() - 0.5) * image.size[0] * 0.5
                shift[1] = (random.random() - 0.5) * image.size[1] * 0.5
            # Resize
            scale = 1
            if random.random() > 0.5:
                scale = random.random() * 0.5 + 0.5
            image = F.affine(image, 0, shift, scale, 0)    # Fills image with to black color
            label = F.affine(label, 0, shift, scale, 0)    # Fills with background label=0


        if self.img_transform is not None:
            image = self.img_transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        # TODO: why ToTensor() works for images but not for labels?
        label = torch.from_numpy(np.array(label, dtype=np.uint8)).long()

        return image, label.long()

    """
    Creates the target label names and associated colors from which the targeted number of classes is calculated.
    To exclude one or several classes from training and validation, just feed them as argument in a comma-separated list form, e.g. 0,1,2.     
    """
    def update_classes(self, excluded=None):

        label_names = [
            "background",
            "skin",
            "hair",
            "beard-mustache",
            "sunglasses",
            "wearable",
            "mouth-mask"]

        label_colors = [
            (0, 0, 0),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (0, 0, 255),
            (255, 0, 255),
            (255, 255, 0)]

        # Create a string code to keep track of the classes that are used
        if excluded:
            used_classes = [idx for idx in range(len(label_colors)) if idx not in excluded]
        else:
            used_classes = range(len(label_colors))
        self.classes_code = ''
        for c in used_classes:
            self.classes_code += str(c)

        # Keep the colors and names of the used classes
        self.label_colors = [color for idx, color in enumerate(label_colors) if idx in used_classes]
        self.label_names  = [name  for idx, name  in enumerate(label_names)  if idx in used_classes]

        # The final number of classes
        self.num_classes  = len(self.label_names)


    def augmentation_folders(self, synth_augmen_types):

        self.synth_augmen_folders    = None
        self.augmentation_folders_id = None
        if synth_augmen_types:
            self.synth_augmen_folders = [synthetic_paths[idx] for idx in synth_augmen_types]
            self.augmentation_folders_id = ''
            for sp in self.synth_augmen_folders:
                self.augmentation_folders_id += sp['name']

    def __ToELFWLabel__(self, data):

        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]

        output = np.zeros((data.shape[0], data.shape[1]))
        for c in range(0,self.num_classes):
            color_mask = (r == self.label_colors[c][0]) & (g == self.label_colors[c][1]) & (b == self.label_colors[c][2])
            output[color_mask] = c

        return output

    """ 
    Private method to get and compute class balancing weights 
    that will be used within the loss function for segmentation
    """
    def __compute_class_balance_weights__(self):

        px_frequencies = np.zeros(self.num_classes)
        im_frequencies = np.zeros(self.num_classes)

        i = 0
        L = len(self.files[self.split])
        for f in self.files[self.split]:

            if not i:
                sys.stdout.flush()
                print('')
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            i+=1
            print((bcolors.GREEN+"    --- Image [%d / %d]"+bcolors.ENDC)%(i, L))

            file_name = f['label']
            image = Image.open(file_name).convert("RGB")
            img = np.array(image)
            img = self.__ToELFWLabel__(img)
            for l in range(0, self.num_classes):
                px = np.sum(img==l)
                # label counts if it is present in the image
                if px > 0:
                    px_frequencies[l] += px
                    im_frequencies[l] += img.size

        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
        print((bcolors.GREEN+"    --- DONE!"+bcolors.ENDC))

        # Mask for indices of appearing classes in the train set
        m   = (px_frequencies>0)
        idx = np.where(m)

        frequencies            = np.divide(px_frequencies[m],  im_frequencies[m])
        pos_median_frequencies = np.divide(np.median(frequencies), frequencies)
        #pos_median_frequencies = np.divide(1, frequencies)

        for l in range(0,len(pos_median_frequencies)):
            self.median_frequencies[idx[0][l]] = pos_median_frequencies[l]


    """ 
    Getter of the class balancing weights 
    """
    def get_class_balance_weights(self):
        return self.median_frequencies.astype(np.float)


class ToELFWLabel(object):

    def __init__(self, label_colors, size):

        self.size = size
        self.label_colors = label_colors

    def __call__(self, input):

        data = np.array(input)
        data = np.reshape(data, (self.size, self.size, 3))
        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]

        output = np.zeros((self.size, self.size))
        for c in range(0, len(self.label_colors)):
            color_mask = (r == self.label_colors[c][0]) & (g == self.label_colors[c][1]) & (b == self.label_colors[c][2])
            output[color_mask] = c

        return Image.fromarray(output)