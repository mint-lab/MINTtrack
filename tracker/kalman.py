from filterpy.kalman import KalmanFilter
import numpy as np
from enum import Enum
import scipy

class TrackStatus(Enum):
    Tentative = 0 # 초기 상태. 객체가 아직 완전히 확인 되지 않음 
    Confirmed = 1 # 객체가 성공적으로 추적
    Coasted   = 2 # 객체가 일시적으로 추적되지 않는 상태. 이후 프레임에서 다시 추적 가능 

# 한 KalmanTrack 인스턴스는 객체 하나(사람 하나) 를 의미한다. 
class KalmanTracker(object):
    count = 1
    def __init__(self, y, R, wx, wy, vmax, w,h,dt=1/30):
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # 관측 행렬
        self.kf.R = R # 관측 노이즈 공분산 행렬 
        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))
    
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.kf.Q = np.dot(np.dot(G, Q0), G.T) # 프로세스 노이즈 공분산 행렬 

        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = y[1]
        self.kf.x[3] = 0

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1 # 해당 추적 객체가 아직 어떤 탐지 결과와도 연결되지 않음을 의미 
        self.w = w
        self.h = h

        self.status = TrackStatus.Tentative

    def update(self, y, R):
        self.kf.update(y,R)

    def predict(self):
        self.kf.predict()
        self.age += 1
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x
    
    def distance(self, y, R):
        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        logdet = np.log(np.linalg.det(S))
        return mahalanobis[0,0] + logdet
   
class KalmanTracker2D(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    count = 1

    def __init__(self, y, R, w,h, dt=1/30):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.eye(2 * ndim, 2 * ndim) # self._motion_mat = kf.F
        for i in range(ndim):
            self.kf.F[i, ndim + i] = dt
        self.kf.H = np.eye(ndim, 2 * ndim) # self._update_mat = kf.H 

        # xyah ... BYTETrack 
        self.kf.x[0] = y[0]
        self.kf.x[1] = y[1]
        self.kf.x[2] = y[2]
        self.kf.x[3] = y[3] 
        
        # 관측 공분산 행렬 
        self.kf.R = R 

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.status = TrackStatus.Tentative

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        x = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        P = np.diag(np.square(std))
        return x, P

    def predict(self, x, P):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        x : ndarray
            The 8 dimensional x vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the x vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 x.

        """
        std_pos = [
            self._std_weight_position * x[3],
            self._std_weight_position * x[3],
            1e-2,
            self._std_weight_position * x[3]]
        std_vel = [
            self._std_weight_velocity * x[3],
            self._std_weight_velocity * x[3],
            1e-5,
            self._std_weight_velocity * x[3]]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        x = np.dot(x, self.kf.F.T)
        P = np.linalg.multi_dot((
            self.kf.F, P, self.kf.F.T)) + Q
        return x, P

    def project(self, mean, P):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        P : ndarray
            The state's P matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and P matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        R = np.diag(np.square(std))

        mean = np.dot(self.kf.H, mean)
        P = np.linalg.multi_dot((
            self.kf.H, P, self.kf.H.T))
        return mean, P + R

    def multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        Q = []
        for i in range(len(mean)):
            Q.append(np.diag(sqr[i]))
        Q = np.asarray(Q)

        mean = np.dot(mean, self.kf.F.T)
        left = np.dot(self.kf.F, covariance).transpose((1, 0, 2))
        P = np.dot(left, self.kf.F.T) + Q

        return mean, P

    def update(self, mean, P, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        P : ndarray
            The state's P matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, P)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(P, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_P = P - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_P

    def distance(self, y, R):
        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        logdet = np.log(np.linalg.det(S))
        return mahalanobis[0,0] + logdet

    #TODO
    def distance(self, y, R, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(y)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
