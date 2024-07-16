import cv2
import os
import argparse

# Dataset directory 
dataset = "../Data/MOT17/test/"
video_folder = "data_video/"

# 이미지 파일이 저장된 디렉토리 경로
parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--seq', type=str, default="MOT17-02-SDP/", help='seq name')
parser.add_argument("--output_video", type=str, default="MOT17_02.avi", help='video name')
parser.add_argument("--fps", type=int, default=25, help='frames per second')
args = parser.parse_args()
video_path = os.path.join(video_folder, args.output_video)

# 필요한 디렉토리가 없을 경우 생성
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

seq = os.path.join(dataset, args.seq, "img1/")
# 이미지 파일 리스트 가져오기 및 정렬
images = [img for img in os.listdir(seq) if img.endswith(".jpg")]
images.sort()

# 첫 번째 이미지로부터 프레임 크기 가져오기
frame = cv2.imread(os.path.join(seq, images[0]))
if frame is None:
    raise ValueError(f"Cannot read the image file {os.path.join(seq, images[0])}.")
height, width, layers = frame.shape

# 비디오 파일 설정 (코덱, 프레임 속도, 크기)
fps = args.fps  # FPS 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for image in images:
    img_path = os.path.join(seq, image)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: Cannot read the image file {img_path}. Skipping...")
        continue
    video.write(frame)

video.release()

print(f"Video saved at {video_path} with {fps} FPS")
