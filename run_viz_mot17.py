from util.run_ucmc import make_args
import argparse
from util.run_viz  import visaulization, top_view, viz_traj
if __name__ == '__main__':

    cam_path = "cam_para/MOT17"
    gmc_path = "gmc/mot17"
    out_path = "output/mot17"
    exp_name = "test"
    dataset  = "MOT17"
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--video', type=str, default="data_video/MOT17_05.avi", help='video file name')
    parser.add_argument('--switch_2D', action='store_true', help="Associate objects in 2D space.") 
    parser.add_argument('--flag_unpro', action='store_true', help="Get xy value using Unprojection") 
    parser.add_argument('--lookup_table', action='store_true', help="Switch lookup table") 
    parser.add_argument('--gmc', type=str, default='gmc/mot17/GMC-MOT17-05.txt')
    parser.add_argument('--det_result', type=str, default='det_results/distortmot17/yolox_x_ablation/MOT17-05-SDP.txt', help='video result text file ')
    parser.add_argument('--output_video', type=str, default='output/distortmot17/val/viz/viz_05.avi', help='result video file name')
    parser.add_argument('--cam_para', type=str, default="cam_para/MOT17/MOT17-05-SDP.txt", help='camera parameter file name')
    parser.add_argument('--wx', type=float, default=5, help='wx')
    parser.add_argument('--wy', type=float, default=5, help='wy')
    parser.add_argument('--vmax', type=float, default=10, help='vmax')
    parser.add_argument('--a', type=float, default=100.0, help='assignment threshold')
    parser.add_argument('--cdt', type=float, default=10.0, help='coasted deletion time')
    parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
    parser.add_argument('--conf_thresh', type=float, default=0.01, help='detection confidence threshold')
    args = parser.parse_args()
    
    # seq 
    seq = "MOT17-05-SDP"
    visaulization(args, seq, "detector/config_mot17_05.json")
    # top_view(args, seq,"detector/config_mot17_13.json")
    # viz_traj(args, seq, "detector/config_mot17_13.json")