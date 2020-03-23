import os
import json
import numpy as np
import cv2
import yaml
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
import ipdb

def draw_pts(img, pts, name):
    cp_img = img.copy()
    print(name, pts)
    for idx, pt in enumerate(pts.astype(np.int32)):
        cv2.circle(cp_img, tuple(pt), 4, (0, 0, 255))
        cv2.putText(cp_img, str(idx), tuple(pt), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))
    cv2.imshow(name, cp_img)
    cv2.waitKey(0)

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
    # display width:522cm, height:313cm
    # PAPER_SIZE = (42, 29.7)
    # DISPLAY_SIZE = (350, 230)
    whole_screen_in_world_cs = np.array([(0, 0, 0),
                                        (522, 0, 0),
                                        (522, 313, 0),
                                        (0, 313, 0)], np.float64)
    meta_file = '../../bst-gaze-data-collection-summary/src/data/cam2/meta.json'
    source_dir = os.path.dirname(meta_file)
    with open(meta_file, 'r') as fid:
        js = json.load(fid)
    screen_marker_corners_in_common_cam_img_plane = np.array(js['screen_marker_corners_in_common_cam_img_plane'], \
                                                        np.float64).reshape(-1, 2)
    screen_marker_corners_in_world_cs = np.array(js['screen_marker_corners_in_world_cs'], np.float64).reshape(-1, 3)
    common_plate_corners_in_common_cam_img_plate = np.array(js['common_plate_corners_in_common_cam_img_plate'], \
                                                    np.float64).reshape(-1, 2)
    common_plate_corners_in_world_cs = np.array(js['common_plate_corners_in_world_cs'], np.float64).reshape(-1, 3)
    common_plate_corners_in_IR_cam_img_plate = np.array(js['common_plate_corners_in_IR_cam_img_plate'], np.float64).reshape(-1, 2)
    #------------------------------
    common_cam_intrainsic_file = os.path.join(source_dir,'../', js['common_cam_intrinsic_data'])
    with open(common_cam_intrainsic_file, 'r') as fid:
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
    ## debug
    img = cv2.imread(os.path.join(source_dir, 'common.png'))
    draw_pts(img, screen_marker_corners_in_common_cam_img_plane, 'screen_marker_corners_in_common_cam_img_plane')
    draw_pts(img, common_plate_corners_in_common_cam_img_plate, 'common_plate_corners_in_common_cam_img_plate')
    img = cv2.imread(os.path.join(source_dir, 'ir.png'))
    draw_pts(img, common_plate_corners_in_IR_cam_img_plate, 'common_plate_corners_in_common_cam_img_plate')
    LEN = 10
    # ----------------------IR camera setings.-------------------------
    ir_cam_intrinsic_file = os.path.join(source_dir, '../', js['ir_cam_intrinsic_data']) 
    with open(ir_cam_intrinsic_file, 'r') as fid:
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
    # pickle_file = '{}.pickle'.format(ir_cam_intrinsic_file)
    pickle_file = os.path.join(source_dir, 'ir_R_T.pickle')
    with open(pickle_file, 'wb') as fid:
        pickle.dump({'R_screen_to_ir_cam' : R_screen_to_ir_cam, \
                    'T_screen_to_ir_cam' : T_screen_to_ir_cam}, \
                        fid) 
        print('write conversion matrix to {}'.format(pickle_file))

    # map the screen points to ir camera 
    whole_screen_in_ir_cs = map_wcs_to_cam_cs(R_screen_to_ir_cam, T_screen_to_ir_cam, whole_screen_in_world_cs)
    screen_pts_in_ir_cs = map_wcs_to_cam_cs(R_screen_to_ir_cam, T_screen_to_ir_cam, screen_marker_corners_in_world_cs)
    common_plate_pts_in_ir_cs = (np.matrix(rot_mat_common_plate_to_ir_cam) * np.matrix(common_plate_corners_in_world_cs).T + tvec_common_plate_to_ir_cam).T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(screen_pts_in_ir_cs[:, 0], screen_pts_in_ir_cs[:, 1], screen_pts_in_ir_cs[:, 2], c='r', marker='o')
    ax.scatter(common_plate_pts_in_ir_cs[:, 0], common_plate_pts_in_ir_cs[:, 1], common_plate_pts_in_ir_cs[:, 2], c= 'y', marker = 'o')
    ax.scatter(whole_screen_in_ir_cs[:, 0], whole_screen_in_ir_cs[:, 1], whole_screen_in_ir_cs[:, 2], c = 'r', marker = 'o')
    LEN = 50
    ax.quiver(0, 0, 0, 1, 0, 0, length=LEN, normalize=False)
    ax.quiver(0, 0, 0, 0, 1, 0, length=LEN, normalize=False)
    ax.quiver(0, 0, 0, 0, 0, 1, length=LEN, normalize=False)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

if __name__ == '__main__':
    # test_rgb_cam_rot_mat()
    run()