import torch
import cv2
import argparse
from tracker.ucmc import UCMCTrack
from detector.detector import Detector
from detector.mapper import Mapper
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.det_res = None  

    def load(self, cam_para_file):
        self.mapper = Mapper(cam_para_file, "MOT17")
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
        
        else:  # Use det_results 
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
    
def main(args):
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

    detector = Detector(flag_unpro=args.flag_unpro)
    detector.load(args.cam_para, det_file=args.det_result, gmc_file=args.gmc, switch_2D=args.switch_2D)
    print("==== Detector Configuration")
    print("* Detector name:", detector.detector_name)
    print("* Mapper name:", detector.mapper_name)
    print("====")
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score, args.switch_2D, detector)

    frame_id = 1
    
    while True:
        if not args.switch_2D:  # 3D
           # Initialize Matplotlib figure and axis
            dpi = 10  # Explicit DPI setting
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax = fig.add_subplot(111)
            ax.set_xlim(-10, 40)  # Set limits for x-axis
            ax.set_ylim(-10, 40)  # Set limits for y-axis
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

        else:  # 2D
            ret, frame_img = cap.read()
            if not ret:
                break
            
            dets = detector.get_dets(frame_id, args.conf_thresh, class_dict['person'])
   
            tracker.update(dets, frame_id)
            for det in dets:
                if det.track_id > 0:
                    cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), 
                                  (int(det.bb_left + det.bb_width), int(det.bb_top + det.bb_height)), 
                                  (0, 255, 0), 2)
                    cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("demo", frame_img)
            video_out.write(frame_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

    if not args.switch_2D:
        plt.close(fig)
    cap.release() 
    if args.switch_2D:
        video_out.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default="data_video/MOT17_04.avi", help='video file name')
parser.add_argument('--switch_2D', action='store_true', help="Associate objects in 2D space.") 
parser.add_argument('--flag_unpro', action='store_true', help="Get xy value using Unprojection") 
parser.add_argument('--gmc', type=str, default='gmc/mot17/GMC-MOT17-04.txt')
parser.add_argument('--det_result', type=str, default='det_results/mot17/yolox_x_ablation/MOT17-04-SDP.txt', help='video result text file ')
parser.add_argument('--output_video', type=str, default='output/MOT17_04_lookup.avi', help='result video file name')
parser.add_argument('--cam_para', type=str, default="cam_para/MOT17/MOT17-04-SDP.txt", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')

args = parser.parse_args()
main(args)
