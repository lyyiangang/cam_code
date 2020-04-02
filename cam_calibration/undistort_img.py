# undistort image. you need calibrate image and get intrinsic matrix first. 
import cv2
import numpy as np
import yaml

def run():
    intrinsic_data = '../../src/data/cam_diweitai_1733_1920x1080.yml'
    img_path = '../../src/data/cam_right/ir.png'
    with open(intrinsic_data, 'r') as fid:
        intrisic_data = yaml.load(fid)
    mxt = np.array(intrisic_data['camera_matrix'])
    dist_coefs = np.array(intrisic_data['dist_coefs'][0])
    img = cv2.imread(img_path)
    cv2.imshow('img', img)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mxt, dist_coefs, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mxt, dist_coefs, None, newcameramtx)
    cv2.imshow('undistort_before_crop', dst)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    print('dst:{}, roi:{}'.format(dst.shape, roi))
    cv2.imshow('undistort_after_crop', dst)
    cv2.waitKey(0)

if __name__ == '__main__':
    run()