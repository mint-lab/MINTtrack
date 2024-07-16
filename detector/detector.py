from .mapper import Mapper
from .gmc import GMCLoader
import numpy as np
import os 

# Detection 클래스 정의, id, bb_left, bb_top, bb_width, bb_height, conf, det_class 포함
class Detection:
    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)

    def get_box(self):
        return [self.bb_left, self.bb_top, self.bb_width, self.bb_height]

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2, self.bb_top+self.bb_height, self.y[0,0], self.y[1,0])

    def __repr__(self):
        return self.__str__()

class DetectionBYTE:
    def __init__(self, id, bb_left=0, bb_top=0, bb_width=0, bb_height=0, conf=0, det_class=0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.xyah = np.array([self.bb_left + self.bb_width/2, 
                     self.bb_top  + self.bb_height/2,
                     float(self.bb_height/self.bb_width),
                     self.bb_height]).reshape((4,1))
        self.y = self.xyah 
        self.R = np.eye(8)

    def get_xyah(self):
        return self.xyah


# Detector 클래스, 텍스트 파일에서 임의의 프레임의 객체 감지 결과를 읽기 위해 사용
class Detector:
    
    def __init__(self, add_noise=False):
        self.seq_length = 0
        self.gmc = None
        self.add_noise = add_noise

    def load(self, cam_para_file, det_file, switch_2D, gmc_file=None):
        self.mapper = Mapper(cam_para_file, "MOT17")
        self.load_detfile(det_file, switch_2D)

        if gmc_file is not None:
            self.gmc = GMCLoader(gmc_file)

    def load_detfile(self, filename, switch_2D):
        
        self.dets = dict()
        # 텍스트 파일 열기
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                # 파일의 각 줄을 읽기
                for line in f.readlines():
                    # 각 줄의 내용을 콤마로 분리
                    line = line.strip().split(',')
                    frame_id = int(line[0])
                    if frame_id > self.seq_length:
                        self.seq_length = frame_id
                    det_id = int(line[1])
                    
                    # 새로운 Detection 객체 생성
                    if switch_2D:  # 2D
                        det = DetectionBYTE(det_id)
                        det.bb_left = float(line[2])
                        det.bb_top = float(line[3])
                        det.bb_width = float(line[4])
                        det.bb_height = float(line[5])
                        det.conf = float(line[6])
                        det.det_class = int(line[7])

                        det.y, det.R = self.mapper.bb2xyah([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
                    else:           # 3D
                        det = Detection(det_id)
                        det.bb_left = float(line[2])
                        det.bb_top = float(line[3])
                        det.bb_width = float(line[4])
                        det.bb_height = float(line[5])
                        det.conf = float(line[6])
                        det.det_class = int(line[7])

                        det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
                    
                    if det.det_class == -1:
                        det.det_class = 0
                    
                    if self.add_noise:
                        if frame_id % 2 == 0:
                            noise_z = 0.5 / 180.0 * np.pi
                        else:
                            noise_z = -0.5 / 180.0 * np.pi
                        self.mapper.disturb_campara(noise_z)
                    
                    if self.add_noise:
                        self.mapper.reset_campara()

                    # det을 딕셔너리에 추가
                    if frame_id not in self.dets:
                        self.dets[frame_id] = []
                    self.dets[frame_id].append(det)
        else:
            pass  

    def get_dets(self, frame_id, conf_thresh=0, det_class=0):
        dets = self.dets[frame_id]
        dets = [det for det in dets if det.det_class == det_class and det.conf >= conf_thresh]
        return dets
    
    def cmc(self, x, y, w, h, frame_id):
        u, v = self.mapper.xy2uv(x, y)
        affine = self.gmc.get_affine(frame_id)
        M = affine[:, :2]
        T = np.zeros((2, 1))
        T[0, 0] = affine[0, 2]
        T[1, 0] = affine[1, 2]

        p_center = np.array([[u], [v - h / 2]])
        p_wh = np.array([[w], [h]])
        p_center = np.dot(M, p_center) + T
        p_wh = np.dot(M, p_wh)

        u = p_center[0, 0]
        v = p_center[1, 0] + p_wh[1, 0] / 2

        xy, _ = self.mapper.uv2xy(np.array([[u], [v]]), np.eye(2))

        return xy[0, 0], xy[1, 0]
    
