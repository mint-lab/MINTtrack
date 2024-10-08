from __future__ import print_function

import numpy as np
from lap import lapjv


from .kalman import KalmanTracker, KalmanTracker2D, TrackStatus

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

class UCMCTrack(object):
    def __init__(self,a1, a2, wx, wy,vmax, max_age, fps, dataset, high_score, switch_2D, detector = None):
        self.wx = wx 
        self.wy = wy
        self.vmax = vmax
        self.dataset = dataset
        self.high_score = high_score
        self.max_age = max_age
        self.a1 = a1
        self.a2 = a2
        self.dt = 1.0/fps

        self.switch_2D = switch_2D 
        # 현재 추적중인 모든 객체의 인덱스 
        self.trackers = []
        # 추적기 상태별 인덱스 
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []

        self.detector = detector

        self.debug_info = {}
        
        # Parameters used in debugging session 
        self.params = {
            'xy': None,
            'R': None,
            'contact_point':None,
            'traj': []
        }

    def update(self,dets,frame_id, show_viz = False):
        
        self.data_association(dets,frame_id,show_viz)
        
        self.associate_tentative(dets,show_viz)
        
        self.initial_tentative(dets,show_viz)
        
        self.delete_old_trackers(show_viz)
        
        self.update_status(dets,show_viz)
                
    def data_association(self,dets,frame_id,show_viz):
        # Separate detections into high score and low score
        detidx_high = []
        detidx_low = []
        for i in range(len(dets)):
            if dets[i].conf >= self.high_score:
                detidx_high.append(i)
            else:
                detidx_low.append(i)

        # Predcit new locations of tracks
        for track in self.trackers:
            track.predict()
            if not self.switch_2D:
                x,y = self.detector.cmc(track.kf.x[0,0],track.kf.x[2,0],track.w,track.h,frame_id)
                track.kf.x[0,0] = x
                track.kf.x[2,0] = y
            else: 
                track.y = self.detector.get_xyah()

        trackidx_remain = [] # Tracklets 중 remains 
        self.detidx_remain = [] # unmatched detections 

        """1. high score matching"""
        # Associate high score detections with tracks
        trackidx = self.confirmed_idx + self.coasted_idx
        num_det = len(detidx_high)
        num_trk = len(trackidx)

        for trk in self.trackers:
            trk.detidx = -1

        if num_det*num_trk > 0: # 첫번째 경우 Hungarian 작동 안되도록 조건문 
            cost_matrix = np.zeros((num_det, num_trk))
            for i in range(num_det):
                det_idx = detidx_high[i] # High score detection 
                for j in range(num_trk):
                    trk_idx = trackidx[j]# confirmed + coasted 
                    cost_matrix[i,j] = self.trackers[trk_idx].distance(dets[det_idx].y, dets[det_idx].R) # Calcluate Mapped Mahalanobis Distance  
                
            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix, self.a1) # self.a1: 매칭 임계값 
            
            for i in unmatched_a:
                self.detidx_remain.append(detidx_high[i])
            for i in unmatched_b:
                trackidx_remain.append(trackidx[i])
            
            for i,j in matched_indices: # Hungarian 결과중 매칭된 결과에 대해서 
                det_idx = detidx_high[i] 
                trk_idx = trackidx[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R) # 매칭된 트랙에 대해서 Kf.update 진행 
                self.trackers[trk_idx].death_count = 0 # 매칭된 트랙에 대해서 death_count 진행 
                self.trackers[trk_idx].detidx = det_idx # 매칭된 트랙과 현재 탐지된 결과의 index 를 매칭 
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id # track_id 과 Det id 매칭 

        else: # 매칭을 진행하지 않아 모두 remains 로 전달 
            self.detidx_remain = detidx_high
            trackidx_remain = trackidx

        """2. Low score matching"""
        # Associate low score detections with remain tracksl
        num_det = len(detidx_low)
        num_trk = len(trackidx_remain)
        if num_det*num_trk > 0:
            # Cost matrix 계산 
            cost_matrix = np.zeros((num_det, num_trk))
            for i in range(num_det):
                det_idx = detidx_low[i]
                for j in range(num_trk):
                    trk_idx = trackidx_remain[j]
                    cost_matrix[i,j] = self.trackers[trk_idx].distance(dets[det_idx].y, dets[det_idx].R)
                
            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a2)
            
            for i in unmatched_b:
                trk_idx = trackidx_remain[i]
                self.trackers[trk_idx].status = TrackStatus.Coasted
                self.trackers[trk_idx].detidx = -1

            for i,j in matched_indices:
                det_idx = detidx_low[i]
                trk_idx = trackidx_remain[j]
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id

    def associate_tentative(self, dets,show_viz):
        num_det = len(self.detidx_remain)
        num_trk = len(self.tentative_idx)

        cost_matrix = np.zeros((num_det, num_trk))
        for i in range(num_det):
            det_idx = self.detidx_remain[i]
            for j in range(num_trk):
                trk_idx = self.tentative_idx[j]
                # TODO: 2D distance 
                cost_matrix[i,j] = self.trackers[trk_idx].distance(dets[det_idx].y, dets[det_idx].R)
            
        matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a1)

        for i,j in matched_indices:
            det_idx = self.detidx_remain[i]
            trk_idx = self.tentative_idx[j]
            self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R)
            self.trackers[trk_idx].death_count = 0
            self.trackers[trk_idx].birth_count += 1 # 잠정적인 추적기가 탐지된 객체와 연속적인 프레임에서 성공적으로 연관된 횟수를 나타낸다. 
            self.trackers[trk_idx].detidx = det_idx
            dets[det_idx].track_id = self.trackers[trk_idx].id
            if self.trackers[trk_idx].birth_count >= 2: # Birth count 가 2 이상이 되면 해당 추적 객체가 최소 두 번 연속해서 탐지되었음을 의미 
                self.trackers[trk_idx].birth_count = 0
                self.trackers[trk_idx].status = TrackStatus.Confirmed # 이 경우 확정 상태로 전환 가능 

        for i in unmatched_b:
            trk_idx = self.tentative_idx[i]
            self.trackers[trk_idx].detidx = -1 # 매칭된 det 결과가 없음을 의미 

        unmatched_detidx = []
        for i in unmatched_a:
            unmatched_detidx.append(self.detidx_remain[i])
        self.detidx_remain = unmatched_detidx

    def initial_tentative(self,dets,show_viz):
        for i in self.detidx_remain:
            if self.switch_2D:
                self.trackers.append(KalmanTracker2D(dets[i].y, dets[i].R, dets[i].bb_width, dets[i].bb_height, self.dt))
            else:   
                self.trackers.append(KalmanTracker(dets[i].y, dets[i].R,self.wx,self.wy,self.vmax, dets[i].bb_width, dets[i].bb_height, self.dt))

            self.trackers[-1].status = TrackStatus.Tentative
            self.trackers[-1].detidx = i
        self.detidx_remain = []

    def delete_old_trackers(self, show_viz):
        i = len(self.trackers) # tracklets 의 개수
        for trk in reversed(self.trackers): # 리스트에서 요소를 삭제할 때 인덱스 문제가 발생하지 않도록 하기 위함 
            trk.death_count += 1
            i -= 1 
            if ( trk.status == TrackStatus.Coasted and trk.death_count >= self.max_age) or ( trk.status == TrackStatus.Tentative and trk.death_count >= 2):
                  self.trackers.pop(i)

    def update_status(self,dets,show_viz):
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []
        for i in range(len(self.trackers)):

            detidx = self.trackers[i].detidx
            if detidx >= 0 and detidx < len(dets):
                self.trackers[i].h = dets[detidx].bb_height
                self.trackers[i].w = dets[detidx].bb_width

            if self.trackers[i].status == TrackStatus.Confirmed:
                self.confirmed_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Coasted:
                self.coasted_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Tentative:
                self.tentative_idx.append(i)

    def print_debug_info(self):
        print('*Debug information')
        print("The length of trajectory: ",self.debug_info["traj"])