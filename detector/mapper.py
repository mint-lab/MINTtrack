import numpy as np
import os
from detector.object_draw_predict import get_cylinder, trim_object, predict_center_from_table 
from detector.config_common import load_camera_config, load_config
import cv2  
import matplotlib.pyplot as plt 

import numpy as np

def getUVError(box): # box 4번째 값이 뭐길래? 
    u = 0.05 * box[3]
    v = 0.05 * box[3]
    if u > 13:
        u = 13
    elif u < 2:
        u = 2
    if v > 10:
        v = 10
    elif v < 2:
        v = 2
    return u, v

def parseToMatrix(data, rows, cols):
    matrix_data = np.fromstring(data, sep=' ')
    matrix_data = matrix_data.reshape((rows, cols))
    return matrix_data

def readKittiCalib(filename, flag_KRT=False):
    # 파일이 존재하는지 확인
    if not os.path.isfile(filename):
        print(f"Calib file could not be opened: {filename}")
        return None, False

    P2 = np.zeros((3, 4))
    R_rect = np.identity(4)
    Tr_velo_cam = np.identity(4)
    KiKo = None

    with open(filename, 'r') as infile:
        for line in infile:
            id, data = line.split(' ', 1)
            if id == "P2:":
                P2 = parseToMatrix(data, 3, 4)
            elif id == "R_rect":
                R_rect[:3, :3] = parseToMatrix(data, 3, 3)
            elif id == "Tr_velo_cam":
                Tr_velo_cam[:3, :4] = parseToMatrix(data, 3, 4)
            KiKo = np.dot(np.dot(P2, R_rect), Tr_velo_cam)

    return KiKo, True

def readCamParaFile(camera_para, flag_KRT=False):
    R = np.zeros((3, 3))
    T = np.zeros((3, 1))
    IntrinsicMatrix = np.zeros((3, 3))
    try:
        with open(camera_para, 'r') as f_in:
            lines = f_in.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "RotationMatrices":
                i += 1
                for j in range(3):
                    R[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            elif lines[i].strip() == "TranslationVectors":
                i += 1
                T = np.array(list(map(float, lines[i].split()))).reshape(-1, 1)
                T = T / 1000
                i += 1
            elif lines[i].strip() == "IntrinsicMatrix":
                i += 1
                for j in range(3):
                    IntrinsicMatrix[j] = np.array(list(map(float, lines[i].split())))
                    i += 1
            else:
                i += 1
    except FileNotFoundError:
        print(f"Error! {camera_para} doesn't exist.")
        return None, False

    Ki = np.zeros((3, 4))
    Ki[:, :3] = IntrinsicMatrix

    Ko = np.eye(4)
    Ko[:3, :3] = R
    Ko[:3, 3] = T.flatten()

    if flag_KRT:
        return IntrinsicMatrix, R, T.flatten(), True
    else:
        KiKo = np.dot(Ki, Ko)
        return Ki, Ko, True
    
class Mapper(object):
    def __init__(self, campara_file, dataset="kitti"):
        self.A = np.zeros((3, 3))
        if dataset == "kitti":
            self.KiKo, self.is_ok = readKittiCalib(campara_file)
            z0 = -1.73
        else:
            self.Ki, self.Ko, self.is_ok = readCamParaFile(campara_file)
            self.KiKo = np.dot(self.Ki, self.Ko)
            z0 = 0

        """
            |th11 th12 th13 * z0 + th14|
            |th21 th22 th23 * z0 + th24|
            |th31 th32 th33 * z0 + th34|
        """
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = z0 * self.KiKo[:, 2] + self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

    def uv2xy(self, uv, sigma_uv):
        if self.is_ok == False:
            return None, None

        # uv1 -> homogeneous form 
        uv1 = np.zeros((3, 1))
        uv1[:2, :] = uv
        uv1[2, :] = 1
        """             
                    |u|   |b1|
            b = A^-1|v| = |b2|                   
                    |1|   |b3|                           
        
        """
        b = np.dot(self.InvA, uv1)
        gamma = 1 / b[2, :]
        C = gamma * self.InvA[:2, :2] - (gamma ** 2) * b[:2, :] * self.InvA[2, :2]
        xy = b[:2, :] * gamma
        sigma_xy = np.dot(np.dot(C, sigma_uv), C.T)
        return xy, sigma_xy

    def xy2uv(self, x, y):
        if self.is_ok == False:
            return None, None
        xy1 = np.zeros((3, 1))
        xy1[0, 0] = x
        xy1[1, 0] = y
        xy1[2, 0] = 1
        uv1 = np.dot(self.A, xy1)
        return uv1[0, 0] / uv1[2, 0], uv1[1, 0] / uv1[2, 0]

    def mapto(self, box):
        uv = np.array([[box[0] + box[2] / 2], [box[1] + box[3]]])
        u_err, v_err = getUVError(box)
        sigma_uv = np.identity(2)
        sigma_uv[0, 0] = u_err * u_err
        sigma_uv[1, 1] = v_err * v_err
        y, R = self.uv2xy(uv, sigma_uv)
        return y, R
    
    def bb2xyah(self, box):
        """box: 
            [bb_left, bb_top, bb_width, bb_height]
        """
        x = box[0] + box[2] / 2
        y = box[1] + box[3] / 2
        a = box[3]/box[2] #h/w 
        h = box[3]
        y = np.array([x,y,a,h]).reshape((4,1))
        
        std = 1 / 20 
        stds = [ std * h, std * h, 1e-1, std * h]
        R = np.diag(np.square(stds))
        
        return y, R 

    def disturb_campara(self, z):
        # z축 회전을 기반으로 회전 행렬 Rz 구성
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

        R = np.dot(self.Ko[:3, :3], Rz)
        # self.Ko를 새 변수 Ko_new에 복사
        Ko_new = self.Ko.copy()
        Ko_new[:3, :3] = R
        self.KiKo = np.dot(self.Ki, Ko_new)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

    def reset_campara(self):
        self.KiKo = np.dot(self.Ki, self.Ko)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

class MapperByUnproject(object):
    
    def __init__(self, campara_file, lookup_table, dataset="kitti"):
    
        if dataset == "kitti":
            pass         
        else: 
            self.K, self.R, self.T, self.is_ok = readCamParaFile(campara_file, flag_KRT=True)
            
            self.Ki = np.zeros((3, 4))
            self.Ki[:, :3] = self.K
            self.Ko = np.eye(4)
            self.Ko[:3, :3] = self.R
            self.Ko[:3, 3] = self.T.flatten()
            self.KiKo = np.dot(self.Ki, self.Ko)
            self.InvK = np.linalg.inv(self.K)
            
            z0 = 0
            self.A = np.zeros((3, 3))
            self.A[:, :2] = self.KiKo[:, :2]
            self.A[:, 2] = z0 * self.KiKo[:, 2] + self.KiKo[:, 3]
            self.InvA = np.linalg.inv(self.A)

            # AICity things 
            self.cam_config = None
            self.lookup_table = lookup_table


    def set_aicity_config(self, config_file):
        satellite, cameras, config = load_config(config_file)
        self.cam_config = cameras[0]

    # def uv2xy_transform(self, uv_point):
    #     """uv 좌표를 xy 좌표로 변환하는 함수"""
    #     uv_undistorted = cv2.undistortPoints(uv_point, self.K, np.array(self.cam_config['distort']))
    #     uv1 = cv2.convertPointsToHomogeneous(uv_undistorted)
    #     uv1 = np.dot(self.K, uv1.reshape((1, 3)).T).T
    #     pt_cam = self.InvK @ uv1.reshape(3, 1)
    #     dir = self.R.T @ pt_cam
    #     pos = -self.R.T @ self.T
    #     scale = (0 - pos[2]) / dir[2]
    #     xyz = pos[:, np.newaxis] + scale * dir
    #     return xyz[:2]  # xy 좌표 반환

    # def uv2xy(self, uv, sigma_uv, z_plane = 0):
    #     if self.is_ok == False:
    #         return None, None

    #     # cylinder 
    #     if self.lookup_table:
    #         delta = predict_center_from_table(uv, self.cam_config['cylinder_table'])
    #         uv = uv + delta[:,np.newaxis]

    #     # Unscented Transform을 사용해 sigma_xy 계산
    #     sigma_points, mean, covariance = self.ut.transform(uv, sigma_uv, self.uv2xy_transform)

    #     xy = mean  # xy 좌표의 평균
    #     sigma_xy = covariance  # xy 좌표의 공분산

    #     return xy, sigma_xy

    # def xy2uv(self, x, y):
    #     if self.is_ok == False:
    #         return None, None
    #     xy1 = np.zeros((3, 1))
    #     xy1[0, 0] = x
    #     xy1[1, 0] = y
    #     xy1[2, 0] = 1
    #     uv1 = np.dot(self.A, xy1)
    #     return uv1[0, 0] / uv1[2, 0], uv1[1, 0] / uv1[2, 0]

    def localize_point(self, uv, K, distort=None, R=np.eye(3), T=np.zeros((3, 1)), polygons={}, planes=[]):
        '''Calculate 3D location (unit: [meter]) of the given point (unit: [pixel]) with the given camera configuration'''
        # Make a ray aligned to the world coordinate
        ori = R.T
        pos = -R.T @ T.squeeze()

        # Undistort point
        uv_undistort = cv2.undistortPoints(np.array(uv, dtype=K.dtype), K, distort).flatten()
        r = ori @ np.append(uv_undistort, 1) # A ray with respect to the world coordinate
        scale = np.linalg.norm(r)
        r = r / scale

        # Get a plane if 'pt' exists inside of any 'polygons'
        n, d = np.array([0, 0, 1]), 0

        # Calculate distance and position on the plane
        denom = n.T @ r
        # if np.fabs(denom) < 1e-6: # If the ray 'r' is almost orthogonal to the plane norm 'n' (~ almost parallel to the plane)
        #     return None, None
        distance = -(n.T @ pos + d) / denom
        # r_c = ori.T @ (np.sign(distance) * r)
        # if r_c[-1] <= 0: # If the ray 'r' stretches in the negative direction (negative Z)
        #     return None, None
        # X = Camera  position + k * ray
        xy = pos + distance * r

        return xy[0:2].reshape(2,1)
    
    def unscented_transform_point(self, uv, sigma_uv, K, distort=None, R=np.eye(3), T=np.zeros((3, 1)), alpha=1e-3, kappa=0):
        '''ref: https://dibyendu-biswas.medium.com/extended-kalman-filter-a5c3a41b2f80'''
        n = uv.shape[0]  # Dimension of the input (2D point)

        # Calculate lambda
        lambda_ = alpha**2 * (n + kappa) - n

        # Create sigma points
        sigma_points = np.zeros((n * 2 + 1, n))
        sigma_points[0] = uv.flatten()
        sqrt_cov = np.linalg.cholesky((n + lambda_) * sigma_uv)
        for i in range(n):
            sigma_points[i + 1] = uv.flatten() + sqrt_cov[i]
            sigma_points[n + i + 1] = uv.flatten() - sqrt_cov[i]

        # Transform sigma points using the localization function
        transformed_points = np.array([self.localize_point(sig_point, K, distort, R, T).flatten() for sig_point in sigma_points])

        # Calculate new mean and covariance
        weights_mean = np.full((2 * n + 1,), 1 / (2 * (n + lambda_)))
        weights_mean[0] = lambda_ / (n + lambda_)
        new_mean = np.dot(weights_mean, transformed_points)

        # Calculate new covariance
        covariance_new = np.zeros((2, 2))
        for i in range(2 * n + 1):
            diff = transformed_points[i] - new_mean
            covariance_new += weights_mean[i] * np.outer(diff, diff)

        return covariance_new

    def uv2xy(self, uv, sigma_uv):
        # distort = np
        xy = self.localize_point(uv, self.K, np.array(self.cam_config['distort']), self.R, self.T)
        sigma_xy = self.unscented_transform_point(uv, sigma_uv, self.K, np.array(self.cam_config['distort']), self.R, self.T)
        return xy, sigma_xy
    
    def xy2uv(self, x, y):
        '''Convert the given 3D point to pixel coordinates with the given camera configuration'''
        
        # Squeeze the position vector
        points_3D = np.array([[x, y, 0]], dtype='float32') 
        distort_coeffs = np.array(self.cam_config['distort'])

        rvec, _ = cv2.Rodrigues(self.R)
        tvec = self.T.squeeze()

        uv, _ = cv2.projectPoints(points_3D, rvec, tvec, self.K, distort_coeffs)

        return uv[0][0][0], uv[0][0][1]

    def mapto(self, box):
        uv = np.array([[box[0] + box[2] / 2], [box[1] + box[3]]])
        u_err, v_err = getUVError(box)
        sigma_uv = np.identity(2)
        sigma_uv[0, 0] = u_err * u_err
        sigma_uv[1, 1] = v_err * v_err
        y, R = self.uv2xy(uv, sigma_uv)
        
        return y, R
    
    def bb2xyah(self, box):
        """box: 
            [bb_left, bb_top, bb_width, bb_height]
        """
        x = box[0] + box[2] / 2
        y = box[1] + box[3] / 2
        a = box[3]/box[2] #h/w 
        h = box[3]
        y = np.array([x,y,a,h]).reshape((4,1))
        
        std = 1 / 20 
        stds = [ std * h, std * h, 1e-1, std * h]
        R = np.diag(np.square(stds))
        
        return y, R 
    
    def disturb_campara(self, z):
        # z축 회전을 기반으로 회전 행렬 Rz 구성
        Rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])

        R = np.dot(self.Ko[:3, :3], Rz)
        # self.Ko를 새 변수 Ko_new에 복사
        Ko_new = self.Ko.copy()
        Ko_new[:3, :3] = R
        self.KiKo = np.dot(self.Ki, Ko_new)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

    def reset_campara(self):
        self.KiKo = np.dot(self.Ki, self.Ko)
        self.A[:, :2] = self.KiKo[:, :2]
        self.A[:, 2] = self.KiKo[:, 3]
        self.InvA = np.linalg.inv(self.A)

    def test_uv2xy(self):
        # Generate ground truth (GT)
        theta = np.linspace(0, 2 * np.pi, 100)
        u = 500 + 100 * np.cos(theta)
        v = 500 + 100 * np.sin(theta)

        uv = np.column_stack((u, v))

        # # uv2xy and xy2uv
        # new_uv = np.zeros_like(uv)
        # for i in range(len(uv)):
        #     n_xy, _ = self.uv2xy(uv[i].reshape((2, 1)), np.identity(2) * 0)  # Reshape to (2, 1) and pass sigma_uv as identity
        #     x, y = n_xy.flatten()  # Flatten the output for compatibility
        #     n_uv = self.xy2uv(x, y)
        #     new_uv[i] = n_uv

        # uv2xy and xy2uv new
        new_uv = np.zeros_like(uv)
        for i in range(len(uv)):
            n_xy, _ = self.uv2xy(uv[i].reshape((2, 1)), np.identity(2))  # Reshape to (2, 1) and pass sigma_uv as identity
            x, y = n_xy.flatten()  # Flatten the output for compatibility
            n_uv = self.xy2uv(x, y)
            new_uv[i] = n_uv
        # Plotting
        plt.figure(figsize=(9, 6))
        plt.plot(u, v, color="red", label="GT", linewidth = 2)
        plt.plot(new_uv[:, 0], new_uv[:, 1], color="blue", label="Mapped", linestyle='--', linewidth = 2)

        # Adjust the limits based on the data
        all_x = np.concatenate((u, new_uv[:, 0]))
        all_y = np.concatenate((v, new_uv[:, 1]))
        plt.xlim([np.min(all_x) - 50, np.max(all_x) + 50])  # Extend limits by a margin
        plt.ylim([np.min(all_y) - 50, np.max(all_y) + 50])  # Extend limits by a margin

        plt.xlabel('u')
        plt.ylabel('v')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.title("2D Circle Plot in u-v Coordinates")
        plt.show()


