
import cv2
import argparse
import os,cv2
import argparse

from tracker.ucmc import UCMCTrack
from detector.detector import Detector
from detector.mapper import Mapper
from detector.object_draw import get_cylinder, draw_cylinder, localize_point
from detector.config_common import load_config
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2, self.bb_top+self.bb_height, self.y[0,0], self.y[1,0])

    def __repr__(self):
        return self.__str__()

class DetectorDemo:
    def __init__(self):
        self.seq_length = 0
        self.gmc = None
        self.det_res = None  

    def load(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
    
        det_results = 'det_results/'
        _, dataset_name, seq_name = cam_para_file.split("/")
        if 'mot17' in dataset_name.lower():
            det_results = det_results + 'mot17/bytetrack_x_mot17/' + seq_name
        else:
            det_results = det_results + 'mot20/' + seq_name 

        with open(det_results, 'r') as f:
            self.det_res = f.readlines()

    def get_dets(self, img, conf_thresh=0, det_classes=[0]):
        dets = []

        # Convert frame from BGR to RGB (because OpenCV uses BGR format)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    
        det_id = 0
        cls_id = 0
        for results in self.det_res:
            frame_seq, id, bb_left, bb_top, w, h, conf, x, y, z = results.split(',')
            w = float(w)
            h = float(h)
            conf = float(conf)
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue

            # Create a new Detection object
            det = Detection(det_id)
            det.bb_left = float(bb_left)
            det.bb_top = float(bb_top)
            det.bb_width = float(w)
            det.bb_height = float(h)
            det.conf = float(conf)
            det.det_class = int(cls_id)
            det.y, det.R = self.mapper.mapto([det.bb_left, det.bb_top, det.bb_width, det.bb_height])
            det_id += 1
            dets.append(det)

        return dets
    
def visaulization(args, seq, config_file):
    plt.rcParams['figure.max_open_warning'] = 0
    class_dict = {"person": 0}
    cap = cv2.VideoCapture(args.video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # Open a cv2 window with specified height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)

    if args.switch_2D:
        cv2.resizeWindow("demo", width, height)

    # Det
    detector = Detector(flag_unpro=args.flag_unpro, lookup_table=args.lookup_table)
    detector.load(args.cam_para, det_file=args.det_result, gmc_file=args.gmc, switch_2D=args.switch_2D)

    # Trk
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, args.switch_2D, detector)
    frame_id = 1
    
    # AICity
    sat, cams, _ = load_config(config_file)
    cam = cams[0]
    cylinder_shape=(0.3, 1.6)
    object_color = (255, 0, 0)

    while True:
        ret, frame_img = cap.read()

        if not ret:
            break
        
        dets = detector.get_dets(frame_id, args.conf_thresh, class_dict['person'])

        tracker.update(dets, frame_id)
        for det in dets:

            if det.track_id > 0:
                # bbox 
                cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), 
                            (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)), 
                            (0, 255, 0), 2)
                
                # ID 
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Lookup Table
                _ , _ = localize_point((int(det.bb_left + det.bb_width/2), int(det.bb_top + det.bb_height)), cam['K'], cam['distort'], cam['ori'], cam['pos'], cam['polygons'], sat['planes'])
                cylinder, _ = get_cylinder((int(det.bb_left + det.bb_width/2), int(det.bb_top + det.bb_height)), *cylinder_shape, sat, cam)
                draw_cylinder(frame_img, cylinder, object_color)

        cv2.imshow("demo", frame_img)
        video_out.write(frame_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Ensure the correct reshaping by adjusting dimensions
        frame_img = frame_img.reshape((int(height), int(width), 3))
        
        cv2.imshow("demo", frame_img)
        video_out.write(frame_img)
        frame_id +=1

    cap.release() 
    video_out.release()
    cv2.destroyAllWindows()


def top_view(args, seq, config_file):
    plt.rcParams['figure.max_open_warning'] = 0
    class_dict = {"person": 0}
    cap = cv2.VideoCapture(args.video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # Open a cv2 window with specified height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)

    if args.switch_2D:
        cv2.resizeWindow("demo", width, height)

    detector = Detector(flag_unpro=args.flag_unpro, lookup_table=args.lookup_table)
    detector.load(args.cam_para, det_file=args.det_result, gmc_file=args.gmc, switch_2D=args.switch_2D)

    print("==== Detector Configuration")
    print("* Detector name:", detector.detector_name)
    print("* Mapper name:", detector.mapper_name)
    print("====")
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, args.switch_2D, detector)

    frame_id = 1
    
    while True:
        # Initialize Matplotlib figure and axis
        dpi = 50  # Explicit DPI setting
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(-10, 20)  # Set limits for x-axis
        ax.set_ylim(-10, 20)  # Set limits for y-axis
        ax.set_aspect('equal')

        frame_img = np.ones((width, height, 3), dtype=np.uint8) * 255  # White background
        dets = detector.get_dets(frame_id, args.conf_thresh, class_dict['person'])
        tracker.update(dets, frame_id)
        for det in dets:
            if det.track_id > 0:
                x, y = det.y[0, 0], det.y[1, 0]
                
                # Plot the position
                ax.plot(x, y, 'ro')
                ax.text(x, y, f'ID: {det.track_id}', fontsize=12, color='red')

                # Calculate and plot the covariance ellipse
                eigvals, eigvecs = np.linalg.eig(det.R[:2, :2])
                order = eigvals.argsort()[::-1]  # Sort eigenvalues in descending order
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]

                # Compute angle for ellipse
                angle = np.arctan2(*eigvecs[:, 0][::-1]) * 180 / np.pi

                # Ellipse width and height
                ell_width, ell_height = 2 * np.sqrt(eigvals)  # 2 * sqrt of eigenvalues for confidence interval

                # Create an ellipse and add to plot
                ellipse = Ellipse((x, y), ell_width, ell_height, angle=angle, edgecolor='blue', facecolor='none')
                ax.add_patch(ellipse)

        # Convert Matplotlib figure to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        frame_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8) # 여기서 왜 줄지?  여이전에서는 frame_img.shape = (1920, 1080, 3) 인데 여기서는 (240000, ) 이디

        # Ensure the correct reshaping by adjusting dimensions
        frame_img = frame_img.reshape((int(height), int(width), 3))
        
        cv2.imshow("demo", frame_img)
        video_out.write(frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id +=1

def viz_traj(args, seq, config_file):
    plt.rcParams['figure.max_open_warning'] = 0
    class_dict = {"person": 0}
    cap = cv2.VideoCapture(args.video)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

    # Open a cv2 window with specified height and width
    cv2.namedWindow("demo", cv2.WINDOW_NORMAL)

    if args.switch_2D:
        cv2.resizeWindow("demo", width, height)

    detector = Detector(flag_unpro=args.flag_unpro, lookup_table=args.lookup_table)
    detector.load(args.cam_para, det_file=args.det_result, gmc_file=args.gmc, switch_2D=args.switch_2D)

    print("==== Detector Configuration")
    print("* Detector name:", detector.detector_name)
    print("* Mapper name:", detector.mapper_name)
    print("====")
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, args.switch_2D, detector)

    frame_id = 1
    specific_track_id = 2# 특정 트랙 ID를 추적하기 위한 파라미터
    trajectory = []  # 특정 트랙 ID의 위치를 저장할 리스트
    
    while True:
        
        # Initialize Matplotlib figure and axis
        dpi = 50  # Explicit DPI setting
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(-10, 20)  # Set limits for x-axis
        ax.set_ylim(-10, 20)  # Set limits for y-axis
        ax.set_aspect('equal')

        frame_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        dets = detector.get_dets(frame_id, args.conf_thresh, class_dict['person'])
        tracker.update(dets, frame_id)
        for det in dets:
            if det.track_id > 0:
                x, y = det.y[0, 0], det.y[1, 0]
            
                if det.track_id == specific_track_id:
                    trajectory.append((x, y))

        # 특정 트랙 ID의 전체 경로 시각화
        if len(trajectory) > 1:
            trajectory_np = np.array(trajectory)
            ax.plot(trajectory_np[:, 0], trajectory_np[:, 1], 'b-', linewidth=2)  # 파란색 선으로 경로 시각화

        # Convert Matplotlib figure to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        frame_img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        
        # Ensure the correct reshaping by adjusting dimensions
        frame_img = frame_img.reshape((int(height), int(width), 3))
        
        cv2.imshow("demo", frame_img)
        video_out.write(frame_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        frame_id +=1
if __name__ == "__main__":

    from run_ucmc import run_ucmc, make_args

    det_path = "det_results/mot17/yolox_x_ablation"
    cam_path = "cam_para/MOT17"
    gmc_path = "gmc/mot17"
    out_path = "output/mot17"
    exp_name = "val"
    dataset = "MOT17"
    args = make_args()
    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)

    # seq 
    seq = "MOT17-04-SDP"
    visaulization(args, seq, "config_mot17_04.json")
