import sys, os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from models import GCN, ResnetFCN, DeepLabV3
from utils import check_mkdir, bcolors
from elfw import ELFWDataSet

# Model type: fcn, gcn, or deeplab
model_type = 'fcn'

# I/O
checkpoints_folder  = "../checkpoints"
input_folder        = "../Datasets/elfw/elfw_01_basic/faces"
output_folder       = "../deploy"

# Input From File
f = open("../Datasets/elfw/elfw_01_basic/elfw_set_00.txt", "r")

test_names = []
for line in f:
    # for some reason it's also loading the \n at the end of each line
    if line[-1:] == '\n':
        test_names.append(line[:-1])
    else:
        test_names.append(line)

# -----------------------------------------------------------------

# Dataset: just wanting some configuration params
dataset = ELFWDataSet(split='test',excluded_classes=[6])

# -----------------------------------------------------------------
# Model loading

if model_type == "fcn":
    model = torch.nn.DataParallel(ResnetFCN(dataset.num_classes))
elif model_type == "gcn":
    gcn_levels = 3
    model = torch.nn.DataParallel(GCN(dataset.num_classes, gcn_levels))
elif model_type == "deeplab":
    model = torch.nn.DataParallel(DeepLabV3(dataset.num_classes))
else:
    print('Model type not found.')
    exit(-1)

# -----------------------------------------------------------------

def main():

    print(bcolors.RED    + "Checkpoints folder: " + checkpoints_folder + bcolors.ENDC)
    print(bcolors.YELLOW + "Deploy folder: "      + output_folder      + bcolors.ENDC)

    check_mkdir(output_folder)

    # List all checkpoint files in the folder
    checkpoints_files = os.listdir(checkpoints_folder)

    # Make predictions for all checkpoints in the deploy folder
    for checkpoint in checkpoints_files:

        print(bcolors.GREEN + "   >> Deploying with " + checkpoint + "..." + bcolors.ENDC)
        checkpoint_filename = os.path.join(checkpoints_folder, checkpoint)

        model.load_state_dict(torch.load(checkpoint_filename))
        model.cuda()
        model.eval()

        # -----------------------------------------------------------------
        # Pass forward

        with torch.no_grad():

            for i, image_name in enumerate(test_names):

                image_name_path = os.path.join(input_folder, image_name + ".jpg")
                image = Image.open(image_name_path).convert("RGB")
                img = dataset.img_transform(image)
                img = Variable(img).cuda().unsqueeze(0)
                scores = model(img)         # first image in the batch
                label_probs = F.log_softmax(scores[0], dim=0).cpu().detach().numpy()

                # -----------------------------------------------------------------
                # Composite

                # a = 0.3  # the smaller the more intense the blending is (more greenish)
                # composite = np.array(image)
                rgb = np.zeros((dataset.target_size, dataset.target_size, 3))
                labels = np.argmax(label_probs, axis=0)

                for l in range(len(label_probs)):
                    indexes = labels == l
                    for c in range(3):
                        rgb[:, :, c][indexes] = dataset.label_colors[l][c]
                    # composite[:, :, c][indexes] = (1 - label_probs[l][indexes]) * composite[:, :, c][indexes] + (a * composite[:, :, c][indexes] + (1 - a) * label_colors[l][c]) * label_probs[l][indexes]

                # -----------------------------------------------------------------
                # Save

                comp = Image.fromarray(rgb.astype('uint8'))
                output_file = os.path.join(output_folder, image_name + "_" + checkpoint[:-4] + ".png")
                comp.save(output_file)

                # -----------------------------------------------------------------
                # Console output

                if i == 0:
                    sys.stdout.flush()
                    print('')
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')

                print((bcolors.BLUE + "    --- [%d / %d] Deployed image " + output_file + bcolors.ENDC) % (i, len(test_names)))

            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            print((bcolors.GREEN+"    --- DONE!"+bcolors.ENDC))

if __name__ == '__main__':
    main()
