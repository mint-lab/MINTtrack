from tracker.kalman import KalmanTracker 
from detector.detector import *

if __name__ == "__main__":
    # Import synthetic GT
    det_file = os.path.join("det_results/synthetic", "circle.txt")
    # Initialize detector
    detector = Detector()
    detector.load(cam_para=None, det_file, gmc_file=None, switch_2D = False)
    # Initialize Kalman Tracker 
    tracker = KalmanTracker()
    
    tracker.predict()