#!/bin/bash

# # MOT17 bytetrack 
# sequence_numbers=("01" "03" "06" "07" "08" "12" "14")
# # 루프 시작
# for i in "${sequence_numbers[@]}"
# do
#     python demo.py --video "data_video/MOT17_${i}.avi" --gmc "gmc/mot17/GMC-MOT17-${i}.txt" --det_result "det_results/mot17/bytetrack_x_mot17/MOT17-${i}-SDP.txt" --use_cmc 1 --output_video "output/MOT17_${i}_output.avi" --cam_para "cam_para/MOT17/MOT17-${i}-SDP.txt"
# done


# mot17 yolox 
sequence_numbers=("02" "04" "05" "09" "10" "11" "13")
# 루프 시작
for i in "${sequence_numbers[@]}"
do
    python demo.py --video "data_video/MOT17_${i}.avi" --gmc "gmc/mot17/GMC-MOT17-${i}.txt" --det_result "det_results/mot17/yolox_x_ablation/MOT17-${i}-SDP.txt" --switch_2D 0 --output_video "output/MOT17_${i}_output_3D.avi" --cam_para "cam_para/MOT17/MOT17-${i}-SDP.txt"
    python demo.py --video "data_video/MOT17_${i}.avi" --gmc "gmc/mot17/GMC-MOT17-${i}.txt" --det_result "det_results/mot17/yolox_x_ablation/MOT17-${i}-SDP.txt" --switch_2D 1 --output_video "output/MOT17_${i}_output_2D.avi" --cam_para "cam_para/MOT17/MOT17-${i}-SDP.txt"
done

# mot20 
# sequence_numbers=("01" "02" "03" "04" "05" "06" "07" "08")
# # 루프 시작
# for i in "${sequence_numbers[@]}"
# do
#     python demo.py --video "data_video/MOT20_${i}.avi" --gmc "gmc/mot20/GMC-MOT20-${i}.txt" --det_result "det_results/mot20/MOT20-${i}-SDP.txt" --use_cmc 1 --output_video "output/MOT20_${i}_output.avi" --cam_para "cam_para/MOT20/MOT20-${i}.txt"
# done
