import torch
from ultralytics import YOLO  # Assuming YOLO class from ultralytics package

def convert_pth_to_pt(pth_file, pt_file):
    # YOLO 모델 초기화 (필요에 따라 모델 클래스를 변경하세요)
    model = YOLO('pretrained/yolov8x.pt')  # 사전 학습된 가중치로 초기화

    # pth.tar 파일 로드 (CPU로 매핑)
    checkpoint = torch.load(pth_file, map_location=torch.device('cpu'))

    # 체크포인트의 키 확인
    print("Checkpoint keys:", checkpoint.keys())

    # 적절한 키로 모델 상태 로드 (여기서는 'model')
    if 'model' in checkpoint:
        model.model.load_state_dict(checkpoint['model'])
    else:
        raise KeyError("Cannot find model state in checkpoint")

    # 모델 상태를 pt 파일로 저장
    torch.save(model.model.state_dict(), pt_file)

# 사용 예제
pth_file = 'pretrained/bytetrack_x_mot17.pth.tar'
pt_file = 'pretrained/bytetrack_x_mot17.pt'
convert_pth_to_pt(pth_file, pt_file)
