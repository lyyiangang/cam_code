import os
import numpy as np
import cv2
import yaml
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
import ipdb

def ray_plane_intersect(ray_vec, ray_start_pt, plane_norm_vec, plane_refrence_pt):
    """ Test the intersection between ray and plane, http://geomalgorithms.com/a06-_intersect-2.html
    """
    ray_vec /= np.linalg.norm(ray_vec)
    ray_vec, plane_norm_vec = np.squeeze(ray_vec), np.squeeze(plane_norm_vec)
    length = np.dot(ray_vec, plane_norm_vec)
    if abs(length) < 1.0e-3:
        return 0, 0
    t = np.dot(np.squeeze(plane_refrence_pt - ray_start_pt), plane_norm_vec) / length
    print('dist:{}'.format(np.dot(ray_start_pt + t * ray_vec - plane_refrence_pt, plane_norm_vec)))
    return ray_start_pt + t * ray_vec, t

def map_wcs_to_cam_cs(r_mat, t_mat, pos_cs):
    assert pos_cs.shape[1] == 3
    return (np.matrix(r_mat) * np.matrix(pos_cs).T + \
                                    np.matrix(t_mat)).T

def run():
    # display width:52.2cm, height:31.3cm
    DISPLAY_SIZE = (52.2, 31.3)
    GLASS_CHESSBOARD_SIZE = (40, 40)
    COMMON_CAM_INTRINSIC_DATA = '../../bst-gaze-data-collection-summary/src/data/cam_LYYovL03_1920x1080_intrinsic.yml'
    IR_CAM_INTRINSIC_DATA = '../../bst-gaze-data-collection-summary/src/data/cam_diweitai_1569_1920x1080.yml'
    # the screen and common plate 2d and 3d coordinate in common camera CS.
    screen_marker_corners_in_common_cam_img_plane = np.array([
                            (888, 105), (1578, 66), (1557, 580), (854, 452),
                            ], dtype = np.float64)

    screen_marker_corners_in_world_cs = np.array([
                            (0, 0, 0),
                            (DISPLAY_SIZE[0], 0, 0),
                            (DISPLAY_SIZE[0], DISPLAY_SIZE[1], 0),
                            (0, DISPLAY_SIZE[1], 0),
                            ], dtype = np.float64)

    common_plate_corners_in_common_cam_img_plate = np.array([
                                                            (194, 486), (614, 414), (638, 716), (282, 851)
                                                            ], dtype = np.float64)

    common_plate_corners_in_world_cs = np.array([
                            (0, 0, 0),
                            (22.5, 0, 0),
                            (22.5, 17.5, 0),
                            (0, 17.5, 0)
                            ], dtype = np.float64)
    # common plate corners in IR camera CS
    common_plate_corners_in_IR_cam_img_plate = np.array([(469, 197), (1135, 19), (1302, 511), (637, 730)], dtype = np.float64)

    #------------------------------

    with open(COMMON_CAM_INTRINSIC_DATA, 'r') as fid:
        intrisic_data = yaml.load(fid)
    common_cam_intrinsic_mat = np.array(intrisic_data['camera_matrix'])
    common_cam_dist_coefs = np.array(intrisic_data['dist_coefs'][0])
    pts_in_common_cams_cs = []
    # ipdb.set_trace()
    _, rvec_screen_to_common_cam, tvec_screen_to_common_cam = cv2.solvePnP(screen_marker_corners_in_world_cs, \
                                                                    screen_marker_corners_in_common_cam_img_plane, \
                                                                    common_cam_intrinsic_mat, \
                                                                    common_cam_dist_coefs,  \
                                                                    flags = cv2.SOLVEPNP_ITERATIVE)
    rmat_screen_to_common_cam, _ = cv2.Rodrigues(rvec_screen_to_common_cam)
    pts_in_common_cams_cs.append((np.matrix(rmat_screen_to_common_cam) * np.matrix(screen_marker_corners_in_world_cs).T + \
                                    np.matrix(tvec_screen_to_common_cam)).T)

    _, rvec_common_plate_to_common_cam, tvec_common_plate_to_common_cam = cv2.solvePnP(common_plate_corners_in_world_cs, \
                                                                                    common_plate_corners_in_common_cam_img_plate, \
                                                                                    common_cam_intrinsic_mat,\
                                                                                    common_cam_dist_coefs, \
                                                                                    flags = cv2.SOLVEPNP_ITERATIVE)
    rmat_common_plate_to_common_cam, _ = cv2.Rodrigues(rvec_common_plate_to_common_cam)
    pts_in_common_cams_cs.append((np.matrix(rmat_common_plate_to_common_cam) * np.matrix(common_plate_corners_in_world_cs).T + \
                                    np.matrix(tvec_common_plate_to_common_cam)).T)
    pts_in_common_cams_cs = np.vstack(pts_in_common_cams_cs)
    print('pts_in_common_cam_cs:{}'.format(pts_in_common_cams_cs))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_in_common_cams_cs[:, 0], pts_in_common_cams_cs[:, 1], pts_in_common_cams_cs[:, 2], c='r', marker='o')
    LEN = 10
    ax.quiver(0, 0, 0, 1, 0, 0, length=LEN, normalize=True)
    ax.quiver(0, 0, 0, 0, 1, 0, length=LEN, normalize=True)
    ax.quiver(0, 0, 0, 0, 0, 1, length=LEN, normalize=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    # ----------------------IR camera setings.-------------------------
    with open(IR_CAM_INTRINSIC_DATA, 'r') as fid:
        intrisic_data = yaml.load(fid)
    # the IR camera can only see the glass plate before it.
    ir_cam_intrinsic_mat = np.array(intrisic_data['camera_matrix'])
    ir_cam_dist_coefs_mat = np.array(intrisic_data['dist_coefs'][0])
    _, rvec_common_plate_to_ir_cam, tvec_common_plate_to_ir_cam = cv2.solvePnP(common_plate_corners_in_world_cs, common_plate_corners_in_IR_cam_img_plate, 
                                                                        ir_cam_intrinsic_mat, ir_cam_dist_coefs_mat, flags = cv2.SOLVEPNP_ITERATIVE)
    rot_mat_common_plate_to_ir_cam, _ = cv2.Rodrigues(rvec_common_plate_to_ir_cam)
    R_screen_to_ir_cam = np.matrix(rot_mat_common_plate_to_ir_cam) * np.linalg.inv(rmat_common_plate_to_common_cam) * np.matrix(rmat_screen_to_common_cam)
    T_screen_to_ir_cam = np.matrix(rot_mat_common_plate_to_ir_cam) * np.linalg.inv(rmat_common_plate_to_common_cam) * (tvec_screen_to_common_cam - \
                                    tvec_common_plate_to_common_cam) + tvec_common_plate_to_ir_cam
    pickle_file = '{}.pickle'.format(os.path.basename(IR_CAM_INTRINSIC_DATA))
    with open(pickle_file, 'wb') as fid:
        pickle.dump({'R_screen_to_ir_cam' : R_screen_to_ir_cam, \
                    'T_screen_to_ir_cam' : T_screen_to_ir_cam}, \
                        fid) 
        print('write conversion matrix to {}'.format(pickle_file))

    # map the screen points to ir camera 
    screen_pts_in_ir_cs= map_wcs_to_cam_cs(R_screen_to_ir_cam, T_screen_to_ir_cam, screen_marker_corners_in_world_cs)
    common_plate_pts_in_ir_cs = (np.matrix(rot_mat_common_plate_to_ir_cam) * np.matrix(common_plate_corners_in_world_cs).T + tvec_common_plate_to_ir_cam).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(screen_pts_in_ir_cs[:, 0], screen_pts_in_ir_cs[:, 1], screen_pts_in_ir_cs[:, 2], c='r', marker='o')
    ax.scatter(common_plate_pts_in_ir_cs[:, 0], common_plate_pts_in_ir_cs[:, 1], common_plate_pts_in_ir_cs[:, 2], c= 'b', marker = 'o')
    ax.quiver(0, 0, 0, 1, 0, 0, length=LEN, normalize=True)
    ax.quiver(0, 0, 0, 0, 1, 0, length=LEN, normalize=True)
    ax.quiver(0, 0, 0, 0, 0, 1, length=LEN, normalize=True)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    # set 
    # img = cv2.imread('./data/rgb_1920x1080_2.jpg')
    # for idx, pt in enumerate(corners_2d_rgb):
    #     pt = pt.astype(np.int32)
    #     cv2.circle(img, tuple(pt), 2, (0, 0, 255))
    #     cv2.putText(img, str(idx), tuple(pt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

if __name__ == '__main__':
    # test_rgb_cam_rot_mat()
    run()