import numpy as np
import os
from detector.object_draw_predict import get_cylinder, trim_object, predict_center_from_table 
from detector.config_common import load_camera_config, load_config

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

    def uv2xy(self, uv, sigma_uv, z_plane = 0):
        if self.is_ok == False:
            return None, None

        # Homogeneous form 
        uv1 = np.zeros((3,1))
        uv1[:2, :] = uv
        uv1[2,  :] = 1

        pt_cam = self.InvK @ uv1 
        dir = self.R.T @ pt_cam
        pos = -self.R.T @ self.T 

        scale = (z_plane - pos[2]) / dir [2]
        xyz = pos[:,np.newaxis] + scale * dir 
        xy = xyz[:2] 

        # cylinder 
        if self.lookup_table:
            delta = predict_center_from_table(xy, self.cam_config['cylinder_table'])
            xy = xy + delta[:,np.newaxis]
       
        # 일단 UCMCtrack sigma 계산법 가져 옴. 
        b = np.dot(self.InvA, uv1)
        gamma = 1 / b[2, :]
        C = gamma * self.InvA[:2, :2] - (gamma ** 2) * b[:2, :] * self.InvA[2, :2]
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
