# undistort image. you need calibrate image and get intrinsic matrix first. 
import os
import cv2
import numpy as np
import yaml
import argparse

# intrinsic_data = '../../src/data/cam_diweitai_1733_1920x1080.yml'
# img_path = '../../src/data/cam_right/ir.png'
parser = argparse.ArgumentParser()
parser.add_argument('--img', default= '', type = str, help = 'image to calibrate')
parser.add_argument('--intrinsic', default = '', type = str, help = 'camera intrinsic data, in yaml formate')

def run(img_path, intrinsic_data):
    assert os.path.exists(img_path) and os.path.exists(intrinsic_data)
    with open(intrinsic_data, 'r') as fid:
        intrisic_data = yaml.load(fid)
    mxt = np.array(intrisic_data['camera_matrix'])
    dist_coefs = np.array(intrisic_data['dist_coefs'][0])
    img = cv2.imread(img_path)
    cv2.imshow('img', img)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mxt, dist_coefs, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mxt, dist_coefs, None, newcameramtx)
    print('dst:{}'.format(dst.shape))
    undistort_img_name = '{}.undistort.png'.format(img_path)
    print('writing undistored image to {}'.format(undistort_img_name))
    cv2.imwrite(undistort_img_name, dst)
    cv2.imshow('undistort_before_crop', dst)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('undistort_after_crop', dst)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args.img, args.intrinsic)