import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 기본 카메라 내부 파라미터 (K)
def get_camera_matrix(fx=1000, fy=1000, cx=960, cy=540):
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return K

# 기본 카메라 외부 파라미터 (R, T)
def get_rotation_matrix(theta_x=0, theta_y=0, theta_z=0):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def get_translation_vector(tx=0, ty=0, tz=0):
    return np.array([tx, ty, tz])

# 2D 좌표를 3D 공간으로 역투영
def unproject_point(K, R, T, point_2d, z_plane=0):
    K_inv = np.linalg.inv(K)
    point_2d_hom = np.array([point_2d[0], point_2d[1], 1])
    point_camera = K_inv @ point_2d_hom
    direction = R @ point_camera
    camera_position = -R.T @ T
    scale = (z_plane - camera_position[2]) / direction[2]
    intersection = camera_position + scale * direction
    return intersection

# 3D 타원 생성
def create_ellipse(center, cov_matrix, num_points=100):
    # 고유값과 고유벡터를 구해 타원의 축을 설정
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    order = eig_vals.argsort()[::-1]
    eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]

    # 타원의 반지름 길이 (표준편차로부터)
    radii = np.sqrt(eig_vals)

    # 타원 점 생성 (3D 확장)
    u = np.linspace(0, 2 * np.pi, num_points)
    ellipse = np.array([radii[0] * np.cos(u), radii[1] * np.sin(u), np.zeros_like(u)])
    ellipse_transformed = eig_vecs @ ellipse
    ellipse_transformed += center[:, None]
    return ellipse_transformed.T


# 불확실성을 표현하는 타원을 시각화
def plot_uncertainty(ax3d, intersections, uncertainties):
    for i, (point, cov) in enumerate(zip(intersections, uncertainties)):
        ellipse_points = create_ellipse(point, cov)
        ax3d.plot(ellipse_points[:, 0], ellipse_points[:, 1], zs=point[2], label=f'Uncertainty Ellipse {i+1}')

# 시각화
def plot_intersection(ax3d, intersections, uncertainties, z_plane=0):
    ax3d.scatter(intersections[:, 0], intersections[:, 1], intersections[:, 2], c='b', marker='^', label='3D Intersections')
    ax3d.set_xlabel('X axis')
    ax3d.set_ylabel('Y axis')
    ax3d.set_zlabel('Z axis')
    plot_uncertainty(ax3d, intersections, uncertainties)

# 클릭 이벤트 처리
def onclick(event):
    if event.inaxes:
        x, y = event.xdata, event.ydata
        print(f"Clicked at: ({x}, {y})")
        clicked_points_2d.append([x, y])
        ax2d.plot(x, y, 'ro')  # 클릭된 지점 표시
        fig2d.canvas.draw()

# Enter 키 누를 때의 이벤트 처리
def onkey(event):
    if event.key == 'enter':
        new_intersections = np.array([unproject_point(K, R, T, pt) for pt in clicked_points_2d])
        all_intersections.extend(new_intersections)
        uncertainty = np.array([np.diag([np.random.uniform(0, 0.1), np.random.uniform(0, 0.1), 0.0])])
        uncertainties.extend(uncertainty)  # 예시로 임의 3D 불확실성을 지정
        plot_intersection(ax3d, np.array(all_intersections), uncertainties)
        fig3d.canvas.draw()

# 테스트
if __name__ == '__main__':
    K = get_camera_matrix()
    R = get_rotation_matrix(np.deg2rad(30), np.deg2rad(-15), np.deg2rad(10))
    T = get_translation_vector(0, 0, 5)
    clicked_points_2d = []
    all_intersections = []
    uncertainties = []
    
    fig2d, ax2d = plt.subplots()
    ax2d.set_xlim(0, 1920)
    ax2d.set_ylim(0, 1080)
    plt.title("Click on the screen, press Enter to visualize in 3D")
    
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.set_title("3D Visualization")
    
    fig2d.canvas.mpl_connect('button_press_event', onclick)
    fig2d.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
