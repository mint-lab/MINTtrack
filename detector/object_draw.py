import numpy as np
import cv2 as cv
import detector.opencx as cx
from detector.object_localize import localize_point, check_polygons
def put_object_on_plane(center_p, object_m, direction, satellite, camera):
    '''Put 3D points (unit: [meter]) on the given point (unit: [pixel]) and direction (unit: [meter])'''
    center_m, _ = localize_point(center_p, camera['K'], camera['distort'], camera['ori'], camera['pos'], camera['polygons'], satellite['planes'])
    if center_m is not None:
        rz = np.array([0, 0, 1])
        plane_idx = check_polygons(center_p, camera['polygons'])
        if (plane_idx >= 0) and (plane_idx < len(satellite['planes'])):
            rz = satellite['planes'][plane_idx][0:3]
            rz = rz / np.linalg.norm(rz)
        rx = np.array([direction[0], direction[1], -(rz[0:2].T @ direction[0:2]) / rz[-1]])
        rx = rx / np.linalg.norm(rx)
        ry = np.cross(rz, rx)
        R = np.vstack((rx, ry, rz)).T
        return center_m + object_m @ R.T
    return None

def get_circle(center_p, radius_m, satellite, camera, offset_m=0, n=16):
    '''Generate a pair of points (unit: [pixel] and [meter]) for a circle on the given point (unit: [pixel])'''
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / n)
    circle = np.array([[radius_m * np.cos(theta), radius_m * np.sin(theta), offset_m] for theta in thetas])
    circle_m = put_object_on_plane(center_p, circle, (1, 0), satellite, camera)
    if circle_m is None:
        return None, None
    circle_p, _ = cv.projectPoints(circle_m, camera['rvec'], camera['tvec'], camera['K'], camera['distort'])
    return circle_p.reshape(-1, 2), circle_m

def get_cylinder(center_p, radius_m, height_m, satellite, camera, offset_m=0, n=32):
    '''Generate a pair of points (unit: [pixel] and unit: [meter]) for a cylinder on the given point (unit: [pixel])'''
    bot_p, bot_m = get_circle(center_p, radius_m, satellite, camera, offset_m, n)
    top_p, top_m = get_circle(center_p, radius_m, satellite, camera, offset_m + height_m, n)
    if (bot_p is None) or (top_p is None):
        return None, None
    return np.vstack((bot_p, top_p)), np.vstack((bot_m, top_m))

def draw_cylinder(image, cylinder_p, color, thickness=2):
    '''Draw a cylinder described as 2D points (unit: [pixel])'''
    half = int(len(cylinder_p)/2)
    bottom, top = cylinder_p[:half], cylinder_p[half:]
    bl_idx, tl_idx = np.argmin(bottom[:,0]), np.argmin(top[:,0])
    br_idx, tr_idx = np.argmax(bottom[:,0]), np.argmax(top[:,0])
    bottom, top = bottom.astype(np.int32), top.astype(np.int32)
    cv.polylines(image, [bottom, top], True, color, thickness)
    cv.line(image, bottom[bl_idx], top[tl_idx], color, thickness)
    cv.line(image, bottom[br_idx], top[tr_idx], color, thickness)

def get_rectangle(center_p, front_m, side_m, direction, satellite, camera, offset_m=0):
    '''Generate a pair of points (unit: [pixel] and [meter]) for a rectangle on the given point (unit: [pixel]) and direction (unit: [meter])'''
    f_half, s_half = front_m / 2, side_m / 2
    rect = np.array([[-s_half, f_half, offset_m], [s_half, f_half, offset_m], [s_half, -f_half, offset_m], [-s_half, -f_half, offset_m]])
    rect_m = put_object_on_plane(center_p, rect, direction, satellite, camera)
    if rect_m is None:
        return None, None
    rect_p, _ = cv.projectPoints(rect_m, camera['rvec'], camera['tvec'], camera['K'], camera['distort'])
    return rect_p.reshape(-1, 2), rect_m

def get_road_direction(pt_p, satellite, camera, offset_m=0, dist_threshold=10):
    '''Find the nearest road direction of the given point (unit: [pixel]) from satellite['roads_data']'''
    pt_m, _ = localize_point(pt_p, camera['K'], camera['distort'], camera['ori'], camera['pos'], camera['polygons'], satellite['planes'])
    if pt_m is not None:
        p = pt_m[:2]
        nearest_dist = dist_threshold
        nearest_idx = -1
        for idx, data in enumerate(satellite['roads_data']):
            p0, v, n = data[:2], data[3:5], data[-1]
            delta = p - p0
            proj_ratio = (delta @ v) / n
            if proj_ratio < 0:
                proj_p = p0
            elif proj_ratio > 1:
                proj_p = p0 + v
            else:
                proj_p = p0 + proj_ratio * v
            dist = np.linalg.norm(proj_p - p)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = idx
        if nearest_idx >= 0:
            return satellite['roads_data'][nearest_idx, 3:5]
    return np.array([1, 0])

def get_cuboid(center_p, front_m, side_m, height_m, direction, satellite, camera, offset_m = 0):
    '''Generate a pair of points (unit: [pixel] and [meter]) for a cuboid on the given point (unit: [pixel]) and direction (unit: [meter])'''
    bot_p, bot_m = get_rectangle(center_p, front_m, side_m, direction, satellite, camera, offset_m)
    top_p, top_m = get_rectangle(center_p, front_m, side_m, direction, satellite, camera, offset_m + height_m)
    if (bot_p is None) or (top_p is None):
        return None, None
    return np.vstack((bot_p, top_p)), np.vstack((bot_m, top_m))

def draw_cuboid(image, cuboid_p, color, thickness=2):
    '''Draw a cuboid described as 2D points (unit: [pixel])'''
    half = int(len(cuboid_p)/2)
    cuboid = cuboid_p.astype(np.int32)
    bottom, top = cuboid[:half], cuboid[half:]
    cv.polylines(image, [bottom, top], True, color, thickness)
    for b, t in zip(bottom, top):
        cv.line(image, b, t, color, thickness)

def test_draw(image_file, config_file, camera_index=0, cylinder_shape=(0.3, 1.6), cuboid_shape=(1.8, 4.5, 1.4)):
    '''Test object drawing on the given image'''

    from config_common import load_config, get_marker_palette

    # A callback function to save the clicked point
    def click_camera_image(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            param[0] = x
            param[1] = y

    # Configure parameters
    object_color = (255, 0, 0)
    cursor_radius = 10
    cursor_color = (0, 0, 255)
    text_offset = (-5, 15)
    text_color = (0, 0, 255)

    # Load configuration and images
    satellite, cameras, _ = load_config(config_file)
    camera = cameras[camera_index]
    camview = cv.imread(image_file)
    if ('polygons' in camera) and (len(camera['polygons']) > 0):
        palette = get_marker_palette(int_type=True, bgr=True)
        for idx, pts in camera['polygons'].items():
            cv.polylines(camview, [pts.astype(np.int32)], True, palette[idx % len(palette)], 2)

    # Get a point and draw an object at the point
    cylinder_mode = True
    click_curr, click_prev = np.array([0, 0]), np.array([0, 0])
    cv.imshow('test_draw', camview)
    cv.setMouseCallback('test_draw', click_camera_image, click_curr)
    while True:
        if not np.array_equal(click_curr, click_prev):
            click_prev = click_curr.copy()

            # Show the point and draw an object at the point
            pt_m, dist_m = localize_point(click_curr, camera['K'], camera['distort'], camera['ori'], camera['pos'], camera['polygons'], satellite['planes'])
            if pt_m is not None:
                camview_viz = camview.copy()

                if cylinder_mode:
                    # Draw a cylinder on the point
                    cylinder, _ = get_cylinder(click_curr, *cylinder_shape, satellite, camera)
                    draw_cylinder(camview_viz, cylinder, object_color)
                else:
                    # Draw a cuboid on the point
                    direction = get_road_direction(click_curr, satellite, camera)
                    cuboid, _ = get_cuboid(click_curr, *cuboid_shape, direction, satellite, camera)
                    draw_cuboid(camview_viz, cuboid, object_color)

                # Draw 'click_curr' as a cross mark
                cv.line(camview_viz, click_curr-[cursor_radius, 0], click_curr+[cursor_radius, 0], cursor_color, 2)
                cv.line(camview_viz, click_curr-[0, cursor_radius], click_curr+[0, cursor_radius], cursor_color, 2)

                # Show 'pt_m' as text
                text = f'XYZ: ({pt_m[0]:.3f}, {pt_m[1]:.3f}, {pt_m[2]:.3f}), Dist: {dist_m:.3f}'
                cx.putText(camview_viz, text, click_curr+text_offset, color=text_color)

                cv.imshow('test_draw', camview_viz)
            else:
                print('* Warning) The clicked point is out of the reference plane.')

        key = cv.waitKey(1)
        if key == ord('\t'): # Tab
            cylinder_mode = not cylinder_mode
        elif key == 27:      # ESC
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    test_draw('detector/data/MOT17_02_screenshot.png', 'detector/config_mot17_02.json')
    test_draw('detector/data/MOT17_04_screenshot.png', 'detector/config_mot17_04.json')
    test_draw('detector/data/MOT17_09_screenshot.png', 'detector/config_mot17_09.json')