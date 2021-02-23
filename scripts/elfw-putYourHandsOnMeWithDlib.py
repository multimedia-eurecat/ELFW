# This code inserts hands from an image to another relative to their respective head poses.
#
# Dlib is used to estimate the head poses, stolen from GitHub:
# KwanHua Lee (lincolnhard) Taiwan, lincolnhardabc@gmail.com
#
# The head pose estimation uses a PnP solver to match between
# the detected facial landmarks and a predefined landmarks set (object_pts).
#
# Note that the model landmarks do not necessarily correspond to the
# actual face, and therefore an accurate matching (regression) is not always guaranteed.
#
# Color correction is performed in the lab color space as explained in:
# P. Shirley et al. 'Color transfer between images' IEEE Corn, vol, 21, pp. 34-41, 2001.
#
# R. Redondo (c) Eurecat 2019

# IMPORTANT NOTE: some hands in Hand Over Faces and Hand2Face datasets present visual artifacts 
# after color correction. They have been identified in the script elfw-cleaner.py.
# Run it immediately after hand data augmentation is done.

import os
import sys
import math
import cv2
import dlib
import argparse
import numpy as np
from imutils import face_utils

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]

D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

label_colors = [
	(  0,  0,  0),
	(  0,255,  0),
	(  0,  0,255),
	(255,255,  0),
	(255,  0,  0),
	(255,  0,255),
	(  0,255,255),
    (  0,  0,  0)]

label_tags = [
	"background",
	"skin",
	"hair",
	"beard-mustache",
	"sunglasses",
	"wearable",
	"mouth-mask",
    "hands"]

dlib_detector = []
dlib_predictor = []

def get_pose_from_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = dlib_detector(gray, 1)

    if not len(face_rects):
        return

    shape = dlib_predictor(gray, face_rects[0])
    shape = face_utils.shape_to_np(shape)

    euler_angles, reprojection = get_head_pose(shape)
    return euler_angles, reprojection, shape, face_rects[0]

def get_head_pose(shape):

    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # Euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return euler_angle, reprojectdst

def get_solid_angle(poseA, poseB):

    # Degrees to radians
    elevationA = math.radians(poseA[0, 0])
    azimuthA   = math.radians(poseA[1, 0])
    rotationA  = math.radians(poseA[2, 0])

    elevationB = math.radians(poseB[0, 0])
    azimuthB   = math.radians(poseB[1, 0])
    rotationB  = math.radians(poseB[2, 0])

    # Spherical to Cartesian
    xA = math.cos(math.radians(azimuthA)) * math.sin(math.radians(elevationA))
    yA = math.sin(math.radians(azimuthA)) * math.sin(math.radians(elevationA))
    zA = math.cos(math.radians(elevationA))

    xB = math.cos(math.radians(azimuthB)) * math.sin(math.radians(elevationB))
    yB = math.sin(math.radians(azimuthB)) * math.sin(math.radians(elevationB))
    zB = math.cos(math.radians(elevationB))

    # Solid spherical angle
    solid_angle = math.acos( xA * xB + yA * yB + zA * zB )

    # Head rotation angle
    solid_rotation = abs(rotationB - rotationA)

    return math.degrees(solid_angle + solid_rotation)


def get_angle_diff_L2(poseA, poseB):

    elevation = poseA[0, 0] - poseB[0, 0]
    azimuth   = poseA[1, 0] - poseB[1, 0]
    rotation  = poseA[2, 0] - poseB[2, 0]

    # L2
    return math.sqrt( elevation * elevation + azimuth * azimuth + rotation * rotation )


def draw_pose(image, reprojection, shape):

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    h, w,_ = image.shape
    for start, end in line_pairs:
        if reprojection[start] >= (0,0) and reprojection[end] >= (0,0) and \
           reprojection[start] <  (w,h) and reprojection[end] <  (w,h):
            cv2.line(image, reprojection[start], reprojection[end], (255, 0, 0))

    return image

def handOverlay(canvas, labels, item, canvas_center, item_center, resize_factor, item_type):

    # Arrays
    canvas_center = np.array(canvas_center).astype(int)
    item_center = np.array(item_center).astype(int)

    # Resize item
    item_center = (item_center * resize_factor).astype(int)
    new_size = (np.array(item.shape[1::-1]) * resize_factor).astype(int)
    item = cv2.resize(item, tuple(new_size))

    # Coordinates
    lt_i = item_center - np.minimum(canvas_center, item_center)
    rb_i = item_center + np.minimum(canvas.shape[1::-1] - canvas_center - 1, item.shape[1::-1] - item_center - 1)

    lt_c = np.maximum(canvas_center - item_center, [0,0])
    rb_c = lt_c + rb_i - lt_i

    # Alpha
    b, g, r, a = cv2.split(item)
    a = a.astype(np.float) / np.max(a)
    aaa = cv2.merge((a,a,a))

    # Crops
    a    =    a[lt_i[1]:rb_i[1], lt_i[0]:rb_i[0]]
    aaa  =  aaa[lt_i[1]:rb_i[1], lt_i[0]:rb_i[0],:3]
    item = item[lt_i[1]:rb_i[1], lt_i[0]:rb_i[0],:3]
    canvas_crop = canvas[lt_c[1]:rb_c[1], lt_c[0]:rb_c[0],:3]
    labels_crop = labels[lt_c[1]:rb_c[1], lt_c[0]:rb_c[0],:3]

    # Blending
    canvas[lt_c[1]:rb_c[1], lt_c[0]:rb_c[0],:3] = canvas_crop * (1-aaa) + item * aaa
    t = label_tags.index(item_type)
    lb, lg, lr = cv2.split(labels_crop)
    lb[a>0] = label_colors[t][0]
    lg[a>0] = label_colors[t][1]
    lr[a>0] = label_colors[t][2]
    labels[lt_c[1]:rb_c[1], lt_c[0]:rb_c[0],:] = cv2.merge((lb,lg,lr))

# -----------------------------------------------------------------------------------------------
# From P. Shirley, et al. 'Color transfer between images' IEEE Corn, vol, 21, pp. 34-41, 2001.

def bgr_to_lab(image_bgr, mask=None):

    M_rgb2lms = [[0.3811, 0.5783, 0.0402],[0.1967, 0.7244, 0.0782],[0.0241, 0.1288, 0.8444]]
    M_lms2lab_1 = [[1.0,1.0,1.0],[1.0,1.0,-2.0],[1.0,-1.0,0.0]]
    M_lms2lab_2 = [[1.0/math.sqrt(3.0),0.0,0.0],[0.0,1.0/math.sqrt(6.0),0.0],[0.0,0.0,1.0/math.sqrt(2.0)]]
    M_lms2lab = np.dot(M_lms2lab_2, M_lms2lab_1)

    h, w, c = image_bgr.shape
    image_bgr = np.transpose( np.reshape(image_bgr, (h * w, c)) )

    rgb = np.flip(image_bgr, 0) / 255.0
    lms = np.dot(M_rgb2lms, rgb)
    lms_log = np.log10(lms + 1E-10)
    lab = np.dot(M_lms2lab, lms_log)

    image_lab = np.reshape( np.transpose(lab), (h, w, c) )

    return image_lab

def lab_to_bgr(image_lab, mask=None):

    M_lms2rgb = [[4.4679, -3.5873, 0.1193],[-1.2186, 2.3809, -0.1624],[0.0497, -0.2439, 1.2045]]
    M_lab2lms_1 = [[1.0,1.0,1.0],[1.0,1.0,-1.0],[1.0,-2.0,0.0]]
    M_lab2lms_2 = [[math.sqrt(3.0)/3.0,0.0,0.0],[0.0,math.sqrt(6.0)/6.0,0.0],[0.0,0.0,math.sqrt(2.0)/2.0]]
    M_lab2lms = np.dot(M_lab2lms_1, M_lab2lms_2)

    h, w, c = image_lab.shape
    image_lab = np.transpose( np.reshape(image_lab, (h * w, c)) )

    lms_log = np.dot(M_lab2lms, image_lab)
    lms = np.power(10, lms_log)
    rgb = np.dot(M_lms2rgb, lms)
    bgr = np.flip(np.clip(rgb, 0, 1), 0) * 255

    image_bgr = np.reshape( np.transpose(bgr), (h, w, c) )

    return image_bgr.astype(np.uint8)

def color_transfer(source_lab, target_lab, source_mask=None):

    if source_mask is not None:
        mask_channels = np.dstack((source_mask, source_mask, source_mask))
        source_lab = np.ma.array(source_lab, mask=(mask_channels == 0))

    source_lab_mean = np.mean(source_lab, axis=(0,1))
    source_lab_std = np.std(source_lab, axis=(0,1))

    target_lab_mean = np.mean(target_lab, axis=(0,1))
    target_lab_std = np.std(target_lab, axis=(0,1))

    s = source_lab.shape[:2]
    lab_std_factor = target_lab_std / source_lab_std

    source_lab -= np.dstack((np.full(s,source_lab_mean[0]), np.full(s,source_lab_mean[1]),  np.full(s,source_lab_mean[2])))
    source_lab *= np.dstack((np.full(s,lab_std_factor[0]),  np.full(s,lab_std_factor[1]),   np.full(s,lab_std_factor[2])))
    source_lab += np.dstack((np.full(s,target_lab_mean[0]), np.full(s,target_lab_mean[1]),  np.full(s,target_lab_mean[2])))

    return source_lab

# -----------------------------------------------------------------------------------------------
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        return False
    else:
        return True

def error(msg):
    print(msg)
    print('Type -h for help')
    sys.exit(0)

def clean_console_line():
    sys.stdout.write('\x1b[1A')
    sys.stdout.write('\x1b[2K')

def main():

    # ----------------------------------------------------------------------
    # Arguments

    ap = argparse.ArgumentParser(prog="elfw-putYourHandsOnMeWithDlib")
    ap.add_argument("-i", "--import",    type=str, help="Import folder with faces and labels to match in /faces and /labels folders, respectively.")
    ap.add_argument("-d", "--dataset",   type=str, help="Path to the hands dataset (Hand2Face or hand_over_face).")
    ap.add_argument("-e", "--export",    type=str, help="Export folder to save faces augmented with hands.")
    ap.add_argument("-t", "--tolerance", type=str, help="Maximum angle threshold to match a pair of faces (positive value in degrees).", default=5)
    args = vars(ap.parse_args())

    angle_tolerance = args['tolerance']

    import_folder = args['import']
    if not import_folder or not os.path.exists(import_folder):
        error('Error: import folder not found')

    import_folder_faces = os.path.join(import_folder,'faces')
    import_folder_labels = os.path.join(import_folder,'labels')
    if not os.path.exists(import_folder_faces) or not os.path.exists(import_folder_labels):
        error('Error: \'/faces\' and \'/labels\' folders not found in the import folder')

    export_folder = args['export']
    if not export_folder:
        error('Error: missing export folder')
        exit(-1)

    export_folder_faces = os.path.join(export_folder, 'faces')
    export_folder_labels = os.path.join(export_folder, 'labels')
    check_mkdir(export_folder)
    check_mkdir(export_folder_faces)
    check_mkdir(export_folder_labels)

    dataset = args['dataset']
    if not os.path.exists(dataset):
        error('Error: dataset folder not found')

    # ----------------------------------------------------------------------
    # Dataloader

    face_names = sorted(os.listdir(import_folder_faces))
    label_names = sorted(os.listdir(import_folder_labels))

    handface_pathnames = []
    handmask_pathnames = []
    if 'hand_over_face' in dataset:
        handfaces_folder = os.path.join(dataset, 'images_original_size')
        handmasks_folder = os.path.join(dataset, 'masks_highres')
        handface_pathnames = [os.path.join(handfaces_folder,f) for f in os.listdir(handfaces_folder)]
        handmask_pathnames = [os.path.join(handmasks_folder,m) for m in os.listdir(handmasks_folder)]
        dataset_label = 'hof'
    elif 'Hand2Face' in dataset:
        handfaces_folder = os.path.join(dataset, 'imgs')
        handmasks_folder = os.path.join(dataset, 'masks')
        handface_pathnames = [os.path.join(handfaces_folder,f) for f in os.listdir(handfaces_folder)]
        handmask_pathnames = [os.path.join(handmasks_folder,m) for m in os.listdir(handmasks_folder)]
        if  'EmoReact' in dataset:
            dataset_label = 'h2f_emo'
        elif 'Web' in dataset:
            dataset_label = 'h2f_web'
        else:
            error('Error: unrecognized Hand2Face dataset type')
    else:
        error('Error: unrecognized dataset type')

    handface_pathnames = sorted(handface_pathnames)
    handmask_pathnames = sorted(handmask_pathnames)

    # ----------------------------------------------------------------------
    # Dlib init

    global dlib_detector, dlib_predictor
    shape_predictor = './dlib/shape_predictor_68_face_landmarks.dat'
    print('\033[1m' + 'Initiating Dlib from ' + shape_predictor + '\033[0m')
    global dlib_detector, dlib_predictor
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(shape_predictor)

    # ----------------------------------------------------------------------
    # Dataset processing

    awkward = 0
    unrecognized = 0
    hints = np.zeros(len(handface_pathnames))

    for i in range(len(handface_pathnames)):

        handface_pathname = handface_pathnames[i]
        _, handface_name = os.path.split(handface_pathname)
        handface_basename, _ = os.path.splitext(handface_name)
        print('Processing ' + '\033[33m' + handface_pathname + '\033[0m')

        # ----------------------------------------------------------------------
        # Get reference pose

        handface = cv2.imread(handface_pathname)
        handface = handface[:, :, :3]  # Leave the alpha channel out
        try:
            handface_angles, reprojection, shape, handface_rect  = get_pose_from_image(handface)
        except:
            unrecognized += 1
            print('Face unrecognized')
            continue

        if abs(handface_angles[0,0]) > 90 or abs(handface_angles[1,0]) > 90 or abs(handface_angles[2,0]) > 90:
            awkward += 1
            print('Skipping awkward pose.')
            continue

        print('Reference pose (elevation,azimuth,rotation) = (%5.2f,%5.2f,%5.2f)' % (handface_angles[0,0],handface_angles[1,0],handface_angles[2,0]))
        # Uncomment the lines below to save the hand-face with pose overlay
        # handface = draw_pose(handface, reprojection, shape)
        # cv2.imwrite(os.path.join(export_folder, handface_name), handface)

        # ----------------------------------------------------------------------
        # Hand masking

        hand_mask = cv2.imread(handmask_pathnames[i], cv2.IMREAD_UNCHANGED)
        if len(hand_mask.shape) > 2:
            hand_mask = hand_mask[:, :, -1]                                                         # take the last channel
        hand_mask = cv2.resize(hand_mask, handface.shape[1::-1], interpolation=cv2.INTER_NEAREST)   # make sure dimensions agree

        handface_lab = bgr_to_lab(handface, hand_mask)

        hw, hh = (handface_rect.right() - handface_rect.left(), handface_rect.bottom() - handface_rect.top())
        handface_center = (handface_rect.left() + hw * 0.5, handface_rect.top() + hh * 0.5)

        # ----------------------------------------------------------------------
        # Iterate over faces in the import folder and paste the reference hands on pose-alike faces

        for t, face_name in enumerate(face_names):

            p_str = '[%d/%d]' % (t,len(face_names))
            print(p_str)

            # ----------------------------------------------------------------------
            # Get pose (on the firstly detected face)

            face = cv2.imread(os.path.join(import_folder_faces, face_name))
            face = face[:, :, :3]  # Leave the alpha channel out
            try:
                angles_face, reprojection, shape, face_rect = get_pose_from_image(face)
            except:
                clean_console_line()
                continue

            # ----------------------------------------------------------------------
            # Difference angle between the reference and match poses

            angle_diff = get_angle_diff_L2(handface_angles, angles_face)
            clean_console_line()

            # Matches
            if angle_diff > angle_tolerance:
                continue
            print('%s Match (%5.2f,%5.2f,%5.2f) %s' % (p_str,angles_face[0,0],angles_face[1,0],angles_face[2,0], face_name))

            # ----------------------------------------------------------------------
            # Hand color correction

            face_roi = face[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right()]
            fh, fw, _ = face_roi.shape
            target_color_roi = [int(fw * 0.2), int(fh * 0.4), int(fw * 0.8), int(fh * 0.6)]
            target_color_roi_lab = bgr_to_lab(face_roi[target_color_roi[1]:target_color_roi[3],target_color_roi[0]:target_color_roi[2]])
            hand_lab = color_transfer(handface_lab, target_color_roi_lab, hand_mask)
            hand = lab_to_bgr(hand_lab, hand_mask)
            hand_bgra = cv2.merge((hand, hand_mask))

            # ----------------------------------------------------------------------
            # Hand overlay

            # image = draw_pose(face, reprojection, shape)
            resize_factor = math.sqrt(fw*fw + fh*fh) / math.sqrt(hw*hw + hh*hh) # ratio between the two face diagonals
            face_center = [face_rect.left() + fw * 0.5, face_rect.top() + fh * 0.5]
            labels = cv2.imread(os.path.join(import_folder_labels, label_names[t]))
            handOverlay(face, labels, hand_bgra, face_center, handface_center, resize_factor, "hands")

            # Uncomment the lines below to draw a rectangle where the target color is measured (mean and std)
            # lefttop = (face_rect.left() + target_color_roi[0], face_rect.top() + target_color_roi[1])
            # rightbottom = (face_rect.left() + target_color_roi[2], face_rect.top() + target_color_roi[3])
            # cv2.rectangle(face, lefttop, rightbottom, (255,255,255), 2)

            face_basename, extension = os.path.splitext(face_name)
            face_filename = face_basename + '-' + dataset_label + '-' + handface_basename + '.jpg'
            labels_filename = face_basename + '-' + dataset_label + '-' + handface_basename + '.png'
            cv2.imwrite(os.path.join(export_folder_faces, face_filename), face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(os.path.join(export_folder_labels, labels_filename), labels)

            hints[i] += 1

    print('Total facehands: %d' % len(handface_pathnames))
    print('Unrecognized facehands: %d' % unrecognized)
    print('Awkward facehands: %d' % awkward)
    print('Total matches: %d' % np.sum(hints))
    print('Usages:')
    print(hints)
    print('\nDone!')

if __name__ == '__main__':
    main()