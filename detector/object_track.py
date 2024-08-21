import numpy as np
import cv2 as cv
import random, time

def get_tracker(name, options={}):
    '''Instantiate a requested tracker'''
    if name.lower() == 'deepsort':
        # Default opts: max_dist=0.2, min_confidence=0.3, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100
        opts = {key : val for key, val in options.items() if key in ['model_path', 'max_dist', 'min_confidence', 'max_iou_distance', 'max_age', 'n_init', 'nn_budget', 'use_cuda']}
        import torch
        from deep_sort_pytorch.deep_sort import DeepSort
        if 'model_path' not in opts:
            opts['model_path'] = 'models/ckpt.t7'
        if 'use_cuda' not in opts:
            opts['use_cuda'] = torch.cuda.is_available()
        return DeepSort(**opts)

def check_polygons(pt, polygons):
    '''Check whether the given point belongs to polygons (index) or not (-1)'''
    if len(polygons) > 0:
        for idx, polygon in polygons.items():
            if cv.pointPolygonTest(polygon, np.array(pt, dtype=np.float32), False) >= 0:
                return idx
    return -1

def test_tracker(video_input,
                 detector_name   = 'YOLOv5',
                 detector_option = {},
                 tracker_name    = 'DeepSORT',
                 tracker_option  = {},
                 tracker_margin  = 1.2,
                 filter_classes  = [0, 2],
                 filter_min_conf = 0.5,
                 filter_rois     = [],
                 start_frame     = 0,
                 frame_offset    = (10, 10),
                 label_offset    = (-8, -24)):
    '''Test an object tracker on the given video'''

    from object_detect import get_detector, draw_2d_rectangle
    import mint.opencx as cx

    # Load the test video
    video = cv.VideoCapture(video_input)
    if start_frame > 0:
        video.set(cv.CAP_PROP_POS_FRAMES, start_frame)

    # Instantiate the object detector
    if start_frame > 0:
        detector_option['start_frame'] = start_frame
    detector = get_detector(detector_name, detector_option)

    # Instantiate the object tracker
    tracker = get_tracker(tracker_name, tracker_option)

    # Post-process ROIs
    if len(filter_rois) > 0:
        filter_rois = {idx: np.array(polygon).astype(np.float32).reshape(-1, 2) for idx, polygon in enumerate(filter_rois)}

    # Test the object detector and tracker
    colors = {}
    frame_total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    while video.isOpened():
        # Get an image
        frame = int(video.get(cv.CAP_PROP_POS_FRAMES))
        ret, img_bgr = video.read()
        if not ret:
            break
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

        # Detect objects
        objects_raw = detector.detect(img_rgb)

        # Remove non-target, low-confidence, and out-of-ROI objects
        objects, objects_remove = [], []
        if objects_raw is not None:
            idx_select, idx_remove = [], []
            for row, obj in enumerate(objects_raw):
                bbox, c, s = obj[0:4], obj[4], obj[5]
                if c in filter_classes:
                    if s > filter_min_conf:
                        bottom = [(bbox[0] + bbox[2]) / 2, bbox[3]]
                        if len(filter_rois) == 0 or check_polygons(bottom, filter_rois) >= 0:
                            idx_select.append(row)
                            continue
                    idx_remove.append(row)
            objects = objects_raw[idx_select]
            objects_remove = objects_raw[idx_remove]

        # Track objects
        elapse = time.time()
        if len(objects) > 0:
            obj_wh = objects[:,2:4] - objects[:,0:2]
            obj_center = objects[:,0:2] + obj_wh / 2
            obj_wh *= tracker_margin # Add margin
            obj_class, obj_score = objects[:,4], objects[:,5]
            tracks = tracker.update(np.hstack((obj_center, obj_wh)), obj_score, obj_class, img_rgb)
            if len(tracks) > 0:
                tracks = tracks.astype(np.float64)
                track_wh = tracks[:,2:4] - tracks[:,0:2]
                track_center = tracks[:,0:2] + track_wh / 2
                track_wh /= tracker_margin # Reduce margin
                tracks[:,0:2] = track_center - track_wh / 2
                tracks[:,2:4] = track_center + track_wh / 2
        else:
            tracker.increment_ages()
            tracks = []
        elapse = time.time() - elapse

        # Visualize removed objects
        if len(objects_remove) > 0:
            for obj_bbox, obj_class, obj_conf in zip(objects_remove[:,0:4], objects_remove[:,4].astype(np.int32), objects_remove[:,5]):
                color = (127, 127, 127)
                draw_2d_rectangle(img_bgr, obj_bbox, color)
                label_pos = obj_bbox[0:2] + label_offset
                cx.putText(img_bgr, f'CL{obj_class}: CF{obj_conf:.2f}', label_pos, color=color)

        # Visualize tracked objects
        if len(tracks) > 0:
            for obj_bbox, obj_id, obj_class in zip(tracks[:,0:4], tracks[:,4].astype(np.int32), tracks[:,5].astype(np.int32)):
                if obj_id not in colors:
                    colors[obj_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw_2d_rectangle(img_bgr, obj_bbox, colors[obj_id])
                label_pos = obj_bbox[0:2] + label_offset
                cx.putText(img_bgr, f'ID{obj_id}, CL{obj_class}', label_pos, color=colors[obj_id])
        cx.putText(img_bgr, f'Frame: {frame}/{frame_total}, FPS: {1/max(elapse,1e-3):.1f} Hz', frame_offset, color=(0, 255, 0), fontScale=0.7)

        # Show the result image (and process the key input)
        cv.imshow('test_tracker', img_bgr)
        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break

    print(f'Last frame: {frame}/{frame_total}')
    video.release()
    cv.destroyAllWindows()



if __name__ == '__main__':
    # test_tracker('data/ETRITestbed/camera4_210812.avi',   tracker_option={'max_age': 30})
    # test_tracker('data/ETRITestbed/camera5_210812.avi',   tracker_option={'max_age': 30})
    # test_tracker('data/ETRITestbed/camera6_210812.avi',   tracker_option={'max_age': 30})
    # test_tracker('data/ETRITestbed/camera7_210812.avi',   tracker_option={'max_age': 30})
    # test_tracker('data/MiraeHall/220722_video.avi',       filter_classes=[2])
    # test_tracker('data/MiraeHall/220722_undistort.avi',   filter_classes=[2])
    # test_tracker('data/ETRIParking/221215_way_in.mkv',    detector_option={'roi_bbox': [222, 186, 1142, 868]}, tracker_option={'min_confidence': 0.2}, filter_classes=[2], filter_min_conf=0.2, filter_rois=[[360, 1047, 0, 1047, 0, 549, 895, 259, 988, 260, 1099, 264, 1132, 271, 1005, 408]])
    # test_tracker('data/ETRIParking/221215_way_out.mkv',   detector_option={'roi_bbox': [222, 186, 1142, 868]}, tracker_option={'min_confidence': 0.2}, filter_classes=[2], filter_min_conf=0.2, filter_rois=[[360, 1047, 0, 1047, 0, 549, 895, 259, 988, 260, 1099, 264, 1132, 271, 1005, 408]])

    # Track objects using 'FileDetector' with loading the detection results
    test_tracker('data/ETRITestbed/camera4_210812.avi',   'data/ETRITestbed/camera4_210812_yolov5.pickle',     tracker_option={'max_age': 30})
    # test_tracker('data/ETRITestbed/camera5_210812.avi',   'data/ETRITestbed/camera5_210812_yolov5.pickle',     tracker_option={'max_age': 30})
    # test_tracker('data/ETRITestbed/camera6_210812.avi',   'data/ETRITestbed/camera6_210812_yolov5.pickle',     tracker_option={'max_age': 30})
    # test_tracker('data/ETRITestbed/camera7_210812.avi',   'data/ETRITestbed/camera7_210812_yolov5.pickle',     tracker_option={'max_age': 30})
    # test_tracker('data/MiraeHall/220722_video.avi',       'data/MiraeHall/220722_video_yolov5.pickle',         filter_classes=[2])
    # test_tracker('data/MiraeHall/220722_undistort.avi',   'data/MiraeHall/220722_undistort_yolov5.pickle',     filter_classes=[2])
    # test_tracker('data/ETRIParking/221215_way_in.mkv',    'data/ETRIParking/221215_way_in_roi_yolov5.pickle',  tracker_option={'min_confidence': 0.2}, filter_classes=[2], filter_min_conf=0.2, filter_rois=[[360, 1047, 0, 1047, 0, 549, 895, 259, 988, 260, 1099, 264, 1132, 271, 1005, 408.]])
    # test_tracker('data/ETRIParking/221215_way_out.mkv',   'data/ETRIParking/221215_way_out_roi_yolov5.pickle', tracker_option={'min_confidence': 0.2}, filter_classes=[2], filter_min_conf=0.2, filter_rois=[[360, 1047, 0, 1047, 0, 549, 895, 259, 988, 260, 1099, 264, 1132, 271, 1005, 408.]])