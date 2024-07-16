import torch.nn as nn
import torch
import os,cv2
import argparse
import torch
from YOLOX.yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from YOLOX.yolox.utils import load_ckpt
from tracker.ucmc import UCMCTrack
from detector.detector import Detector
from detector.mapper import Mapper
import numpy as np

class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
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


    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()

class DetectorDemo:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.det_res = None  

    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        # YOLOX 
        model_name = " "

        if model_name == "pretrained/bytetrack_x_mot17.pth.tar":
            # X
            depth = 1.33
            width = 1.25
            num_classes = 1

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
            head = YOLOXHead(num_classes, width, in_channels=in_channels)
            model = YOLOX(backbone, head)
            ckpt = torch.load(model_name, map_location=self.device)
            model.load_state_dict(ckpt["model"])
            self.model = model.to(self.device)
        
        else: # Use det_results 
            det_results = 'det_results/'
            _, dataset_name, seq_name = cam_para_file.split("/")
            if 'mot17' in dataset_name.lower():
                det_results = det_results + 'mot17/bytetrack_x_mot17/' + seq_name
            else:
                det_results = det_results + 'mot20/'                   + seq_name 

            with open(det_results, 'r') as f:
                self.det_res = f.readlines()

    def get_dets(self, img, conf_thresh = 0,det_classes = [0]):
        
        dets = []

        # 将帧从 BGR 转换为 RGB（因为 OpenCV 使用 BGR 格式）  
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
        det_id = 0
        cls_id = 0
        for results in self.det_res:       
            # conf = box.conf.cpu().numpy()[0]
            # bbox = box.xyxy.cpu().numpy()[0]
            # cls_id  = box.cls.cpu().numpy()[0]
            # w = bbox[2] - bbox[0]
            # h = bbox[3] - bbox[1]

            frame_seq, id, bb_left,bb_top, w, h, conf,x,y,z = results.split(',')
            w = float(w)
            h = float(h)
            conf = float(conf)
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # 新建一个Detection对象
            det = Detection(det_id)
            det.bb_left = float(bb_left)
            det.bb_top = float(bb_top)
            det.bb_width = float(w)
            det.bb_height = float(h)
            det.conf = float(conf)
            det.det_class = int(cls_id)
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det_id += 1

            dets.append(det)

        return dets
    
def main(args):
    class_dict = {"person":0}
    cap = cv2.VideoCapture(args.video)

    # 获取视频的 fps
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # 打开一个cv的窗口，指定高度和宽度
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("demo", width, height)

    detector = Detector()
    detector.load(args.cam_para, det_file=args.det_result, gmc_file=args.gmc, switch_2D=args.switch_2D)

    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, args.use_cmc, args.switch_2D, detector)

    # 循环读取视频帧
    frame_id = 1
    while True:
        ret, frame_img = cap.read()
        if not ret:  
            break
        # dets = detector.get_dets(frame_img, args.conf_thresh, class_list)
        dets = detector.get_dets(frame_id, args.conf_thresh, class_dict['person'])
        tracker.update(dets, frame_id)
        for det in dets:
            # 画出检测框
            if det.track_id > 0:
                cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), (int(det.bb_left+det.bb_width), int(det.bb_top+det.bb_height)), (0, 255, 0), 2)
                # 画出检测框的id
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_id += 1


        # 显示当前帧
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)

        video_out.write(frame_img)
    
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default = "data_video/MOT20_02.avi", help='video file name')
parser.add_argument('--switch_2D', type=bool, default=False, help= "0: associate objects in 2D space, 1: associate objects in 3D space.")
parser.add_argument('--gmc', type=str, default='gmc/mot20/GMC-MOT20-02.txt')
parser.add_argument('--det_result', type=str, default='det_results/mot20/MOT20-02.txt', help = 'video result text file ')
parser.add_argument('--use_cmc', type=bool, default = False, help='Use CMC or not')
parser.add_argument('--output_video', type=str, default='output/MOT20_02_output.avi', help='result video file name')
parser.add_argument('--cam_para', type=str, default = "cam_para/MOT20/MOT20-02.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
args = parser.parse_args()
main(args)

