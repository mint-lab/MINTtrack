import os
from eval.eval import eval


dataset_path = "../Data/MOT17/train"
out_path = "output/distortmot17"
exp_name = "val"

seqmap = os.path.join(out_path,exp_name, "val_seqmap.txt")

# 生成val_seqmap.txt文件
with open(seqmap,"w") as f:
    f.write("name\n")
    f.write("MOT17-02-SDP\n")

HOTA,IDF1,MOTA,AssA = eval(dataset_path, out_path, seqmap, exp_name, 1, False)