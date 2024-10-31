# MINTRACK 

#### Environment
Before you begin, ensure you have the following prerequisites installed on your system:
- Python (3.8 or later)
- PyTorch with CUDA support
- Ultralytics Library
- Download weight file [yolov8x.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) to folder `pretrained`

#### Run the demo

```bash
python demo.py --cam_para demo/cam_para.txt --video demo/demo.mp4
```
The file `demo/cam_para.txt` is the camera parameters estimated from a single image. The code of this tool is released.  For specific steps, please refer to the Get Started.

## 🗼 Pipeline of UCMCTrack
First, the detection boxes are mapped onto the ground plane using homography transformation. Subsequently, the Correlated Measurement Distribution (CMD) of the target is computed. This distribution is then fed into a Kalman filter equipped with the Constant Velocity (CV) motion model and Process Noise Compensation (PNC). Next, the mapped measurement and the predicted track state are utilized as inputs to compute the Mapped Mahalanobis Distance (MMD). Finally, the Hungarian algorithm is applied to associate the mapped measurements with tracklets, thereby obtaining complete tracklets.

![](docs/pipeline.png)

## 🖼️ Visualization of Different Distances
(a) Visualization of IoU on the image plane. IoU fails as there is no intersection between bounding boxes. (b) Visualization of Mapped Mahalanobis Distance (MMD) without Correlated Measurement Distribution (CMD). Incorrect associations occur due to insufficient utilization of distribution information. (c) Visualization of MMD with CMD. Correct associations after using the correlated probability distribution, undergoing a rotation on the ground plane.

![](docs/distance_measure.png)

## 🏃 Benchmark Performance
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot17)](https://paperswithcode.com/sota/multi-object-tracking-on-mot17?p=ucmctrack-multi-object-tracking-with-uniform) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ucmctrack-multi-object-tracking-with-uniform/multi-object-tracking-on-mot20-1)](https://paperswithcode.com/sota/multi-object-tracking-on-mot20-1?p=ucmctrack-multi-object-tracking-with-uniform)

| Dataset    | HOTA | AssA | IDF1 | MOTA | FP     | FN     | IDs   | Frag  |
| ---------- | ---- | ---- | ---- | ---- | ------ | ------ | ----- | ----- |
| MOT17 test | 65.8 | 66.4 | 81.0 | 80.6 | 36,213 | 71,454 | 1,689 | 2,220 |
| MOT20 test | 62.8 | 63.5 | 77.4 | 75.6 | 28,678 | 96,199 | 1,335 | 1,370 |


## 💁 Get Started
