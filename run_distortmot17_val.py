from util.run_ucmc import run_ucmc, make_args

if __name__ == '__main__':

    det_path = "det_results/distortmot17/yolox_x_ablation"
    cam_path = "cam_para/MOT17"
    gmc_path = "gmc/mot17"
    out_path = "output/distortmot17"
    exp_name = "val"
    dataset = "MOT17"
    args = make_args()
    run_ucmc(args, det_path, cam_path, gmc_path, out_path, exp_name,dataset)
