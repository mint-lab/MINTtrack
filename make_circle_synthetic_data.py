import numpy as np 

if __name__ == "__main__":
    with open("det_results/synthetic/circle.txt", "w") as f:
        obj_id = 1  # 객체 ID 설정
        for frame_id in range(1, 300):  # 1부터 시작하여 300 프레임까지
            theta = np.linspace(0, 2 * np.pi, frame_id)
            x_values = 5 + 2 * np.cos(theta)
            y_values = 5 + 2 * np.sin(theta)
            
            for x, y in zip(x_values, y_values):
                f.write(f"{frame_id},{obj_id},{x:.1f},{y:.1f},-1,-1,-1,-1\n")
