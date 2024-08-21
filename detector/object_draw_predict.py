import numpy as np
import cv2 as cv
import pickle
from detector.config_common import load_config
from detector.object_draw import get_cylinder, get_cuboid, draw_cylinder, draw_cuboid, get_road_direction

def trim_object(obj, image_size):
    '''Trim 2D points within the given image size'''
    image_w, image_h = image_size
    obj[(obj[:,0] < 0), 0] = 0
    obj[(obj[:,1] < 0), 1] = 0
    obj[(obj[:,0] >= image_w), 0] = image_w - 1
    obj[(obj[:,1] >= image_h), 1] = image_h - 1

def get_object_bottom_mid(obj):
    '''Get the bottom middle point of the given 2D points'''
    return [(min(obj[:,0]) + max(obj[:,0])) / 2, max(obj[:,1])]

def gen_cylinder_data(satellite, camera, image_size, image_step, cylinder_shape=(0.3, 1.6)):
    '''Generate a lookup table for cylinders for the specific camera'''
    image_w, image_h = image_size
    data = []
    for y in range(0, image_h, image_step):
        for x in range(0, image_w, image_step):
            center = np.array((x, y))
            obj, _ = get_cylinder(center, *cylinder_shape, satellite, camera)
            if obj is not None:
                trim_object(obj, image_size)
                bottom_mid = get_object_bottom_mid(obj)
                delta = center - bottom_mid
                data.append(bottom_mid + delta.tolist())
    return np.array(data)

def save_lookup_table(config_file, image_size=(1920, 1080), image_step=100, save_prefix=''):
    '''Generate and save lookup tables (for cylinders) for multiple cameras to pickle files'''
    satellite, cameras, _ = load_config(config_file)
    for idx, cam in enumerate(cameras):
        data = gen_cylinder_data(satellite, cam, image_size, image_step)
        with open(save_prefix + cam['name'] + '_cylinder.pickle', 'wb') as f:
            pickle.dump(data, f)

def predict_center_from_table(bottom_mid, table, dist_threshold=100):
    '''Predict a foot point using the given lookup table and nearest search'''
    x, y = bottom_mid
    dist = np.fabs(table[:,0] - x) + np.fabs(table[:,1] - y)
    min_idx = np.argmin(dist)
    if dist[min_idx] < dist_threshold:
        return table[min_idx,2:4]
    return np.zeros(2)

def draw_bbox(image, obj_p, color, thickness=2):
    '''Draw a bounding box of the given 2D points'''
    tl = np.array((min(obj_p[:,0]), min(obj_p[:,1])))
    br = np.array((max(obj_p[:,0]), max(obj_p[:,1])))
    cv.rectangle(image, tl.astype(np.int32), br.astype(np.int32), color, thickness)
    return tl, br

def test_table(image_file, config_file, camera_name='camera', cylinder_shape=(0.3, 1.6), cuboid_shape=(1.8, 4.5, 1.4)):
    '''Test a lookup table which predicts a foot point'''

    # A callback function to save the clicked point
    def click_camera_image(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            param[0] = x
            param[1] = y

    # Configure parameters
    object_color = (100, 100, 100)
    cursor_radius = 10
    cursor_color = (0, 0, 255)
    bbox_color = (255, 0, 0)
    predict_radius = 5

    # Load configuration and images
    satellite, cameras, _ = load_config(config_file)
    cam = next(filter(lambda cam: cam['name'] == camera_name, cameras))
    camview = cv.imread(image_file)

    camview_size = camview.shape[0:2][::-1]

    # Get a point and draw an object at the point
    cylinder_mode = True
    click_curr, click_prev = np.array([0, 0]), np.array([0, 0])
    cv.imshow('test_table', camview)
    cv.setMouseCallback('test_table', click_camera_image, click_curr)
    while True:
        if not np.array_equal(click_curr, click_prev):
            click_prev = click_curr.copy()

            # Show the point and draw an object at the point
            camview_viz = camview.copy()
            bottom_mid, delta = click_curr, np.zeros(2)
            if cylinder_mode:
                # Draw a cylinder on the point
                cylinder, _ = get_cylinder(click_curr, *cylinder_shape, satellite, cam)
                if cylinder is not None:
                    trim_object(cylinder, camview_size)
                    bottom_mid = get_object_bottom_mid(cylinder)
                    if 'cylinder_table' in cam:
                        delta = predict_center_from_table(bottom_mid, cam['cylinder_table'])
                    draw_cylinder(camview_viz, cylinder, object_color)
                    draw_bbox(camview_viz, cylinder, bbox_color)
            else:
                # Draw a cuboid on the point
                direction = get_road_direction(click_curr, satellite, cam)
                cuboid, _ = get_cuboid(click_curr, *cuboid_shape, direction, satellite, cam)
                if cuboid is not None:
                    trim_object(cuboid, camview_size)
                    bottom_mid = get_object_bottom_mid(cuboid)
                    if 'cuboid_table' in cam:
                        delta = predict_center_from_table(bottom_mid, cam['cuboid_table'])
                    draw_cuboid(camview_viz, cuboid, object_color)
                    draw_bbox(camview_viz, cuboid, bbox_color)

            # Draw 'click_curr' as a cross mark and the predicted center as a circle
            cv.line(camview_viz, click_curr-[cursor_radius, 0], click_curr+[cursor_radius, 0], cursor_color, 2)
            cv.line(camview_viz, click_curr-[0, cursor_radius], click_curr+[0, cursor_radius], cursor_color, 2)
            center = bottom_mid + delta
            cv.circle(camview_viz, center.astype(np.int32), predict_radius, bbox_color, -1)

            cv.imshow('test_table', camview_viz)

        key = cv.waitKey(1)
        if key == ord('\t'): # Tab
            cylinder_mode = not cylinder_mode
        elif key == 27:      # ESC
            break

    cv.destroyAllWindows()



if __name__ == '__main__':
    # save_lookup_table('detector/config_mot17_02.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_04.json', image_step=10, save_prefix = "detector/data/")
    # save_lookup_table('detector/config_mot17_09.json', image_step=10, save_prefix = "detector/data/")

    test_table('detector/data/MOT17_02_screenshot.png', 'detector/config_mot17_02.json', camera_name='MOT17_02')
    # test_table('detector/data/MOT17_04_screenshot.png', 'detector/config_mot17_04.json', camera_name='MOT17_04')
    # test_table('detector/data/MOT17_09_screenshot.png', 'detector/config_mot17_09.json', camera_name='MOT17_09')