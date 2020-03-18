import numpy as np
import cv2
import yaml
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

PAPER_SIZE, DISPLAY_SIZE = (42, 29.7), (105.5, 59.5)
GLASS_CHESSBOARD_SIZE = (40, 40)

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


def test_rgb_cam_rot_mat():
    """ evaluate the rotation and translation matrix in DMS IR camera coordinate system.
        Background: Since the 3 displays are behind the IR camera, we can't get the display corners' position in camera coordinate system.
        this system trys to solve this problem. the system contains 2 cameras, one RGB camera, it captures 4 * 3 = 12 reference points on 3 displays and 4 reference points on 
        one glass chessboard. The other one is IR camera. it captures the same one glass chessboard and use the same 4 reference points like the RGB camera does.
        this function will try to map the points on displays to IR camera coordinate system.
    """
    #A3 paper size: 42*29.7 cm, display size: 105.5*59.5cm
    corners_2d_rgb = np.array([
                            (456, 143), (728, 74), (753, 276), (490, 358),
                            (744, 72), (1048, 51), (1066, 257), (770, 257),
                            (1523, 39), (1788, 90), (1782, 316), (1527, 249),
                            # glass chessboard
                            (532, 509), (1101, 352), (1253, 894), (725, 1009)
                            ], dtype = np.float64)
    corners_3d_rgb = np.array([
                            # paper corners on left window
                            (DISPLAY_SIZE[0] - PAPER_SIZE[0], 0,               0), 
                            (DISPLAY_SIZE[0],                 0,               0),
                            (DISPLAY_SIZE[0],                 PAPER_SIZE[1], 0),
                            (DISPLAY_SIZE[0] - PAPER_SIZE[0], PAPER_SIZE[1], 0),
                            # paper corners on mid window
                            (0, 0, 0), 
                            (PAPER_SIZE[0], 0, 0),
                            (PAPER_SIZE[0], PAPER_SIZE[1], 0), 
                            (0, PAPER_SIZE[1], 0),
                            #paper corners on right window
                            (0, 0, 0), 
                            (PAPER_SIZE[0], 0, 0),
                            (PAPER_SIZE[0], PAPER_SIZE[1], 0), 
                            (0, PAPER_SIZE[1], 0),
                            #glass chessboard
                            (0, 0, 0),
                            (GLASS_CHESSBOARD_SIZE[0], 0, 0),
                            (GLASS_CHESSBOARD_SIZE[0], GLASS_CHESSBOARD_SIZE[1], 0),
                            (0, GLASS_CHESSBOARD_SIZE[1], 0)
                            ], dtype = np.float64)
    with open('./data/rgb_cam_for_measure_display_coord.yml', 'r') as fid:
        intrisic_data = yaml.load(fid)
    cam_mat_rgb = np.array(intrisic_data['camera_matrix'])
    dis_coefs_rgb = np.array(intrisic_data['dist_coefs'][0])
    # img_size = (1920, 1080)
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_mat_rgb, dis_coefs_rgb, img_size, 1, img_size)

    pts_in_rgb_cs = []
    rot_mats = []
    tvecs = []
    for ii in range(4):
        start_idx = ii * 4
        ret, rvec, tvec = cv2.solvePnP(corners_3d_rgb[start_idx : start_idx + 4, :], corners_2d_rgb[start_idx : start_idx + 4, :], 
                                    cam_mat_rgb, dis_coefs_rgb, flags = cv2.SOLVEPNP_ITERATIVE) #np.zeros((1,4))
        rmat, _ = cv2.Rodrigues(rvec)
        rot_mats.append(rmat)
        tvecs.append(tvec)
        pts_in_rgb_cs.append((np.matrix(rmat) * np.matrix(corners_3d_rgb[start_idx : start_idx + 4]).T + np.matrix(tvec)).T)
    np.savetxt('pts_in_rgb_cam_cs.xyz', np.vstack(pts_in_rgb_cs))

    with open('./data/simulator_IR_cam.yml', 'r') as fid:
        intrisic_data = yaml.load(fid)
    cam_mat_ir = np.array(intrisic_data['camera_matrix'])
    dis_coefs_ir = np.array(intrisic_data['dist_coefs'][0])
    corners_2d_ir = np.array([(1101, 173), (611, 65), (485, 580), (1032, 677)], dtype = np.float64)
    ret, rvec_glass_cb_to_ir_cam, tvec_glass_cb_to_ir_cam = cv2.solvePnP(corners_3d_rgb[3 * 4 : 3 * 4+ 4, :], corners_2d_ir, 
                                                                        cam_mat_ir, dis_coefs_ir, flags = cv2.SOLVEPNP_ITERATIVE)
    rot_mat_glass_cb_to_ir_cam, _ = cv2.Rodrigues(rvec_glass_cb_to_ir_cam)

    def get_display_rot_trans_mat(rot_mat_display_to_rgb_cam, tvec_display_to_rgb_cam):
        rot_mat_glass_cb_to_rgb_cam = rot_mats[3] # glass cs to rgb cs
        tvec_glass_cb_to_rgb_cam = tvecs[3]
        R_d2i = np.matrix(rot_mat_glass_cb_to_ir_cam) * np.linalg.inv(rot_mat_glass_cb_to_rgb_cam) * np.matrix(rot_mat_display_to_rgb_cam)
        T_d2i = np.matrix(rot_mat_glass_cb_to_ir_cam) * np.linalg.inv(rot_mat_glass_cb_to_rgb_cam) * (tvec_display_to_rgb_cam - tvec_glass_cb_to_rgb_cam) + tvec_glass_cb_to_ir_cam
        return R_d2i, T_d2i

    pts_in_ir_cam_cs = []
    # map 3 A3 paper corners to IR camera coordinate system
    disp_rot_trans_mat_dict = {}
    for ii in range(4 * 4):
        window_idx = ii // 4
        R, T = get_display_rot_trans_mat(rot_mats[window_idx], tvecs[window_idx])
        pts_in_ir_cam_cs.append((R * np.matrix(corners_3d_rgb[ii, :]).T + T).T)
        if window_idx not in disp_rot_trans_mat_dict:
            disp_rot_trans_mat_dict[window_idx] = {'R' : R, 'T' : T}
    rot_tran_file_name = './data/disp_rot_trans_mat_dict.npz'
    print('saving {}'.format(rot_tran_file_name))
    np.savez(rot_tran_file_name, disp_rot_trans_mat_dict = disp_rot_trans_mat_dict)
    # np.load('tmp.npz')['disp_rot_trans_mat_dict'].item()['1']
    pts_in_ir_cam_cs = np.vstack(pts_in_ir_cam_cs)
    np.savetxt('pts_in_ir_cam_cs.xyz', pts_in_ir_cam_cs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_in_ir_cam_cs[:, 0], pts_in_ir_cam_cs[:, 1], pts_in_ir_cam_cs[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    img = cv2.imread('./data/rgb_1920x1080_2.jpg')
    for idx, pt in enumerate(corners_2d_rgb):
        pt = pt.astype(np.int32)
        cv2.circle(img, tuple(pt), 2, (0, 0, 255))
        cv2.putText(img, str(idx), tuple(pt), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    


if __name__ == '__main__':
    test_rgb_cam_rot_mat()