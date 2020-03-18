#!/usr/bin/env python
#usage: python calibrate_wide_angle_camera.py --input myvideo.avi
#usage: python calibrate_wide_angle_camera.py --input '../../imgs/*.*g'
# support video and images.
import numpy as np
import cv2
import os
import argparse
import yaml
import pickle
import ipdb
from glob import glob
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate camera using a video of a chessboard or a sequence of images.')
    parser.add_argument('--input', help='input video file or glob mask')
    parser.add_argument('--out_config', default = './cal_result.yml', help='output calibration yaml file')
    parser.add_argument('--debug_dir',  help='path to directory where images with detected chessboard will be written',
                        default='./debug_dir')
    parser.add_argument('-s', '--square_size', help='squre size(cm)', default=2.5)
                        
    parser.add_argument('-c', '--corners', help='output corners file', default=None)
    parser.add_argument('-fs', '--framestep', help='use every nth frame in the video', default=5, type=int)
    parser.add_argument('-fc', help = 'fisheye video?', default = False, type = bool)
    args = parser.parse_args()
    PATTERN_SIZE = (9, 6)
    MAX_FRAMES = 100
    FLIP_Y = False
    # ipdb.set_trace()
    if '*' in args.input:
        source = glob(args.input)
        assert len(source) > 0, 'get empty list'
        random.shuffle(source)
    else:
        source = cv2.VideoCapture(args.input)

    pattern_points = np.zeros((np.prod(PATTERN_SIZE), 3), np.float32)
    pattern_points[:, :2] = np.indices(PATTERN_SIZE).T.reshape(-1, 2)
    pattern_points *= args.square_size 

    obj_points = []
    img_points = []
    h, w = 0, 0
    i = -1
    while True:
        i += 1
        if len(img_points) > MAX_FRAMES:
            break
        if isinstance(source, list):
            # glob
            if i == len(source):
                break
            img = cv2.imread(source[i])
        else:
            # cv2.VideoCapture
            retval, img = source.read()
            if not retval:
                break
            if i % args.framestep != 0:
                continue
        print('Searching for chessboard in frame ' + str(i) + '...')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if FLIP_Y:
            img = cv2.flip(img, flipCode =0)
        h, w = img.shape[:2]
        found, corners = cv2.findChessboardCorners(img, PATTERN_SIZE, cv2.CALIB_CB_ADAPTIVE_THRESH )
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            img_points.append(corners.reshape(1, -1, 2))
            obj_points.append(pattern_points.reshape(1, -1, 3))

        if args.debug_dir:
            img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img_chess, PATTERN_SIZE, corners, found)
            cv2.imwrite(os.path.join(args.debug_dir, '%04d.png' % i), img_chess)
        if not found:
            print('not found square corners from input')
            continue

        print('ok')

    if args.corners:
        with open(args.corners, 'wb') as fw:
            pickle.dump(img_points, fw)
            pickle.dump(obj_points, fw)
            pickle.dump((w, h), fw)
        
    print('\nPerforming calibrate camera...')
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print("RMS:{}".format(rms))
    print("camera matrix:{}\n".format(camera_matrix))
    print("distortion coefficients:{}".format(dist_coefs.ravel()))


    calibration = {'rms': rms, 'camera_matrix': camera_matrix.tolist(), 'dist_coefs': dist_coefs.tolist() }
    with open(args.out_config, 'w') as fw:
        yaml.dump(calibration, fw)
    print('parameters are saved to {}, you can use undistort.py to calibrate your image now'.format(args.out_config))