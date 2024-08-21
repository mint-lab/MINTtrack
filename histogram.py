import matplotlib.pyplot as plt 
from detector.detector import Detector
from tracker.ucmc import UCMCTrack
from tracker.kalman import TrackStatus
from eval.interpolation import interpolate
import os,time
import argparse
from util.run_ucmc import make_args

def run_histogram(args, det_path = "det_results/mot17/yolox_x_ablation",
                   cam_path = "cam_para/mot17",
                   gmc_path = "gmc/mot17",
                   out_path = "output/mot17",
                   exp_name = "val",
                   dataset  = "MOT17"):
    # args 
    seq_name = args.seq
    a1 = args.a
    a2 = args.a
    high_score = args.high_score
    conf_thresh = args.conf_thresh
    fps = args.fps
    cdt = args.cdt
    wx = args.wx
    wy = args.wy
    vmax = args.vmax
    switch_2D = args.switch_2D

    # os 
    eval_path = os.path.join(out_path,exp_name)
    orig_save_path = os.path.join(eval_path,seq_name)
    if not os.path.exists(orig_save_path):
        os.makedirs(orig_save_path)

    if dataset == "MOT17":
        det_file = os.path.join(det_path, f"{seq_name}-SDP.txt")
        cam_para = os.path.join(cam_path, f"{seq_name}-SDP.txt")
        result_file = os.path.join(orig_save_path,f"{seq_name}-SDP.txt")

    elif dataset == "MOT20":
        det_file = os.path.join(det_path, f"{seq_name}.txt")
        cam_para = os.path.join(cam_path, f"{seq_name}.txt")
        result_file = os.path.join(orig_save_path,f"{seq_name}.txt")

    gmc_file = os.path.join(gmc_path, f"GMC-{seq_name}.txt")

    print('Detection File: ',det_file)
    print('Parameter File: ',cam_para)

    flag_unpro = args.flag_unpro
    add_cam_noise = args.add_cam_noise
    lookup_table = args.lookup_table

    detector = Detector(flag_unpro=flag_unpro, 
                        add_noise=add_cam_noise, 
                        lookup_table=lookup_table)
    detector.load(cam_para, det_file, gmc_file, switch_2D=args.switch_2D)
    print(f"seq_length = {detector.seq_length}")

    tracker = UCMCTrack(a1, a2, wx, wy, vmax, cdt, fps, dataset, high_score, switch_2D, detector)

    t1 = time.time()

    tracklets = dict()
    with open(result_file,"w") as f:
        for frame_id in range(1, detector.seq_length + 1):
            dets = detector.get_dets(frame_id, conf_thresh)
            tracker.update(dets,frame_id)



if __name__ == '__main__':

    det_path = "det_results/mot17/yolox_x_ablation"
    cam_path = "cam_para/MOT17"
    gmc_path = "gmc/mot17"
    out_path = "output/mot17"
    exp_name = "val"
    dataset = "MOT17"
    args = make_args()
    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)
