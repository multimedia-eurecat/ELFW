# This code runs dlib-based head pose estimation.
#
# Modified from: KwanHua Lee (lincolnhard) Taiwan, lincolnhardabc@gmail.com
# By: R. Redondo (c) Eurecat 2019

import math
import cv2
import numpy as np
import dlib
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


def get_head_pose(shape):

    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def main():

    shape_predictor = './facedetectors/dlib/shape_predictor_68_face_landmarks.dat'
    print('\033[1m' + 'Initiating Dlib from ' + shape_predictor + '\033[0m')
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor(shape_predictor)

    cap = cv2.VideoCapture(0)

    frame_count = 0

    while True:

        frame_count += 1
        ret, frame = cap.read()

        if not ret or cv2.waitKey(1) & 0xFF == 27: # esc to exit
            cap.release()
            break

        # Face detection: requires grayscale, rect contains as many bounding boxes as faces detected
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = dlib_detector(gray, 1)

        # If no detection, then continue
        if not len(face_rects):
            cv2.imshow("Dlib head pose estimation", frame)
            continue

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = dlib_predictor(gray, face_rects[0])
        shape = face_utils.shape_to_np(shape)

        reprojectdst, euler_angle = get_head_pose(shape)

        # Convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # (x, y, w, h) = face_utils.rect_to_bb(face_rects[0])
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw (x,y)-landmark coordinates
        for (x, y) in shape:
          cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


        for start, end in line_pairs:
            cv2.line(frame, reprojectdst[start], reprojectdst[end], (255, 0, 0))

        elevation = euler_angle[0, 0]
        azimuth   = euler_angle[1, 0]
        rotation  = euler_angle[2, 0]
        # x = math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation))
        # y = math.sin(math.radians(azimuth)) * math.cos(math.radians(elevation))
        # z = math.sin(math.radians(elevation))
        solid_angle = math.degrees(math.acos(math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation)))) # simplified dot product [1,0,0] \dot [x,y,z]

        cv2.putText(frame, "Elevation: "    + "{:7.2f}".format(-elevation),  (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)
        cv2.putText(frame, "Azimuth: "      + "{:7.2f}".format(azimuth),     (20, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)
        cv2.putText(frame, "Rotation: "     + "{:7.2f}".format(rotation),    (20, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)
        cv2.putText(frame, "Solid Angle: "  + "{:7.2f}".format(solid_angle), (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), thickness=2)

        # Show the output image with the face detections + facial landmarks
        cv2.imshow("Dlib head pose estimation", frame)

    cv2.destroyAllWindows()
    print('Done!')

if __name__ == '__main__':
    main()