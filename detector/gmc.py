import numpy as np

# GMCLoader 클래스 정의, GMC 파일을 로드하기 위해 사용
class GMCLoader:
    def __init__(self, gmc_file):
        self.gmc_file = gmc_file
        self.affines = dict()
        self.load_gmc()

    def load_gmc(self):
        # self.gmc_file 파일 열기
        with open(self.gmc_file, 'r') as f:
            # 파일의 각 줄을 읽기
            for line in f.readlines():
                # 각 줄의 내용을 공백 또는 탭으로 분리
                line = line.strip().split()
                frame_id = int(line[0]) + 1

                # 새로운 2x3 행렬 생성
                affine = np.zeros((2, 3))
                # 각 줄의 내용을 float 타입으로 변환
                affine[0, 0] = float(line[1])
                affine[0, 1] = float(line[2])
                affine[0, 2] = float(line[3])
                affine[1, 0] = float(line[4])
                affine[1, 1] = float(line[5])
                affine[1, 2] = float(line[6])

                self.affines[frame_id] = affine

    def get_affine(self, frame_id):
        return self.affines[frame_id]
